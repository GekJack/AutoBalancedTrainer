#pragma once
#include <torch/torch.h>
#include <iostream>
#include <thread>            
#include <mutex>              
#include <condition_variable> 
#include <atomic>
#include <memory>
#include "Types.hpp"
#include "Worker.hpp"
#include "DataLoader.hpp"
#include "DeviceProfiler.hpp"
#include "LoadBalancer.hpp"

// LocalTrainer acts as the "Master" orchestrator in our Master-Worker architecture.
// It manages the thread pool, splits the incoming data batches, synchronizes 
// the CPU and GPU workers, and performs the final gradient aggregation.
class LocalTrainer {
public:
	using LogCallback = std::function<void(const TrainingMetrics&)>;
	using SyncCallback = std::function<std::pair<std::vector<float>, EpochAssignment>(const std::vector<float>&, double, int)>;

	void SetSyncCallback(SyncCallback callback) {
		network_sync_callback_ = std::move(callback);
	}
private:
	int net_start_idx = 0;
	int net_end_idx = 0;
	LogCallback log_callback = nullptr; // Function pointer for logging training progress and metrics
	SyncCallback network_sync_callback_ = nullptr;
	LoadBalancer balancer;
	float ema_loss = 0.0f;

	// Thread management for isolated execution
	std::thread thread_gpu;
	std::thread thread_cpu;
	std::mutex mtx; // Protects shared variables during thread signaling

	// Condition variables used for "Barrier Synchronization"
	std::condition_variable cv_start; // Master uses this to wake up workers
	std::condition_variable cv_done;  // Workers use this to tell Master they are finished

	std::unique_ptr<Worker> worker_gpu;
	std::unique_ptr<Worker> worker_cpu;
	TrainingConfig config;

	// State machine to control thread lifecycles
	enum class State { Idle, Working, Stop } state = State::Idle;
	int workers_done = 0; // Counter to track how many workers finished the current batch
	bool gpu_has_work = false;
	bool cpu_has_work = false;

	// Flags for training flow control
	bool is_profiler = false; // Ensures we only run the initial hardware warm-up/profiling once
	bool is_hybrid = false;   // True if both CPU and GPU are available and configured

	// Shared memory buffers to pass sliced batch data to the workers
	torch::Tensor shared_gpu_data, shared_gpu_target;
	torch::Tensor shared_cpu_data, shared_cpu_target;

	double time_gpu = 0.0;
	double time_cpu = 0.0;
	std::shared_ptr<IModel> model_gpu_ = nullptr;
	std::shared_ptr<IModel> model_cpu_ = nullptr;
	bool random_seed = false;
	float current_gpu_loss = 0.0f;
	float current_cpu_loss = 0.0f;

	// Copies the updated model parameters from the GPU (Source) to the CPU (Target).
	// This ensures both workers start the next batch with identical network knowledge.
	void SyncWeights(std::shared_ptr<IModel> source, std::shared_ptr<IModel> target) {
		torch::NoGradGuard no_grad; // Disable gradient tracking during copying to save memory
		auto source_params = source->named_parameters();
		auto target_params = target->named_parameters();

		for (auto& param : target_params) {
			param.value().copy_(source_params[param.key()]);
		}
	}

private:
	// The core mathematical merge function.
	// It takes the gradients calculated by the CPU, moves them to the GPU memory, 
	// and calculates a weighted average: (GPU_Grad * GPU_Ratio) + (CPU_Grad * CPU_Ratio).
	void AggregateGrad(float gpu_weight) {
		torch::NoGradGuard no_grad;
		float cpu_weight = 1.0f - gpu_weight;
		auto gpu_params = worker_gpu->GetModel()->named_parameters();
		auto cpu_params = worker_cpu->GetModel()->named_parameters();

		for (auto& item : gpu_params) {
			const std::string& name = item.key();
			auto& gpu_p = item.value();
			auto& cpu_p = cpu_params[name];

			if (gpu_p.grad().defined() && cpu_p.grad().defined()) {
				// Move CPU gradient over the PCIe bus to VRAM
				auto cpu_grad_on_gpu = cpu_p.grad().to(torch::kCUDA);

				// Perform weighted averaging directly in VRAM
				gpu_p.grad().copy_(gpu_p.grad() * gpu_weight + cpu_grad_on_gpu * cpu_weight);
			}
		}
	}

	// Initializes the persistent background threads for CPU and GPU.
	// We create threads once and put them to sleep, rather than re-creating them 
	// every batch, which prevents massive OS overhead.
	void InitThreadPool() {
		// GPU THREAD
		thread_gpu = std::thread([this]() {
			torch::DeviceGuard device_guard(torch::Device(torch::kCUDA, 0));
			while (true) {
				std::unique_lock<std::mutex> lock(mtx);
				// Sleep until Master gives the start signal OR tells the thread to stop
				cv_start.wait(lock, [this] { return gpu_has_work && (state == State::Working || state == State::Stop); });
				if (state == State::Stop) break; // Exit thread safely
				gpu_has_work = false;
				lock.unlock(); // Unlock so Master/CPU can use the mutex while we compute

				auto start = std::chrono::high_resolution_clock::now();
				auto gpu_loss = worker_gpu->ComputeGrad(shared_gpu_data, shared_gpu_target);
				torch::cuda::synchronize(); // Wait for actual hardware completion
				auto end = std::chrono::high_resolution_clock::now();
				time_gpu = std::chrono::duration<double, std::milli>(end - start).count();

				// Report back to Master
				lock.lock();
				current_gpu_loss = gpu_loss;
				workers_done++;
				if (workers_done == 2) cv_done.notify_one(); // Wake up Master if we are the last to finish
			}
		});

		// CPU THREAD
		thread_cpu = std::thread([this]() {
			while (true) {
				std::unique_lock<std::mutex> lock(mtx);
				cv_start.wait(lock, [this] { return cpu_has_work && (state == State::Working || state == State::Stop); });
				if (state == State::Stop) break;
				cpu_has_work = false;
				lock.unlock();

				auto start = std::chrono::high_resolution_clock::now();
				auto cpu_loss = worker_cpu->ComputeGrad(shared_cpu_data, shared_cpu_target);
				auto end = std::chrono::high_resolution_clock::now();
				time_cpu = std::chrono::duration<double, std::milli>(end - start).count();

				lock.lock();
				current_cpu_loss = cpu_loss;
				workers_done++;
				if (workers_done == 2) cv_done.notify_one();
			}
			});
	}

	// The main dispatch function for hybrid mode.
	// It slices the batch, wakes up the workers, waits for them, and merges results.
	template<typename BatchType>
	float ExecuteHybridBatch(const BatchType& batch, std::function<void(double, int)> sync_hook = nullptr) {
		// 1. Handle fallback cases if balancer disabled CPU
		if (balancer.IsGpuOnly()) {
			if (!balancer.ShouldTestCPU()) {
				auto loss = worker_gpu->TrainBatch(batch.data.to(torch::kCUDA), batch.target.to(torch::kCUDA), sync_hook);
				return loss;
			}
			else {
				std::cout << "[LocalTrainer] Test CPU..." << std::endl;
				SyncWeights(worker_gpu->GetModel(), worker_cpu->GetModel());
				balancer.ForceHybrid();
			}
		}

		float current_ratio = balancer.GetRatio();
		int total_size = batch.data.size(0);

		// Prevent slicing errors on extremely small batches
		if (total_size < 2) {
			std::cout << "[LocalTrainer] Batch too small for hybrid execution. Running on GPU only." << std::endl;
			auto loss = worker_gpu->TrainBatch(batch.data.to(torch::kCUDA), batch.target.to(torch::kCUDA));
			SyncWeights(worker_gpu->GetModel(), worker_cpu->GetModel());
			return loss;
		}

		// 2. Slice the batch according to the balancer's ratio
		int split_index = std::clamp(static_cast<int>(total_size * current_ratio), 1, total_size - 1);

		// Non-blocking copy to target memory (CPU/GPU)
		shared_gpu_data = batch.data.slice(0, 0, split_index).to(torch::kCUDA, true);
		shared_gpu_target = batch.target.slice(0, 0, split_index).to(torch::kCUDA, true);
		shared_cpu_data = batch.data.slice(0, split_index, total_size).to(torch::kCPU);
		shared_cpu_target = batch.target.slice(0, split_index, total_size).to(torch::kCPU);

		// 3. Wake up workers
		{
			std::lock_guard<std::mutex> lock(mtx);
			state = State::Working;
			workers_done = 0;
			gpu_has_work = true;
			cpu_has_work = true;
		}
		cv_start.notify_all();

		// 4. Master goes to sleep until both workers are done
		{
			std::unique_lock<std::mutex> lock(mtx);
			cv_done.wait(lock, [this] { return workers_done == 2; });
			state = State::Idle;
		}

		// 5. Finalize the batch: merge gradients, update optimizer, sync weights
		AggregateGrad(current_ratio);
		if (sync_hook) {
			double max_time = std::max(time_gpu, time_cpu);
			sync_hook(max_time, batch.data.size(0));
		}
		{
			torch::DeviceGuard device_guard(torch::Device(torch::kCUDA, 0));
			worker_gpu->ApplyGrad();
		}
		SyncWeights(worker_gpu->GetModel(), worker_cpu->GetModel());

		// 6. Report execution times to the balancer for the next batch
		balancer.Update(time_gpu, time_cpu);
		float combined_loss = (current_gpu_loss * split_index + current_cpu_loss * (total_size - split_index)) / (float)total_size;
		return combined_loss;
	}

	// Flattens the multi-dimensional gradient tensors into a single contiguous 1D array.
	// This is required for sending the gradients over the network via ZeroMQ.
	std::vector<float> FlattenAndReduceLocalGradients(float current_gpu_ratio = 1.0f) {
		std::vector<float> flat_grads;

		auto target_params = (config.type == TrainingConfig::DeviceType::CPU)
			? worker_cpu->GetModel()->parameters()
			: worker_gpu->GetModel()->parameters();

		for (const auto& param : target_params) {
			if (param.grad().defined()) {
				auto grad_cpu = param.grad().cpu().contiguous().view({ -1 });
				const float* data_ptr = grad_cpu.data_ptr<float>();
				flat_grads.insert(flat_grads.end(), data_ptr, data_ptr + grad_cpu.numel());
			}
			else {
				// If 0 images were processed, gradients don't exist in memory.
				// We MUST pad the array with zeros so the payload size matches the Master's expectations!
				flat_grads.insert(flat_grads.end(), param.numel(), 0.0f);
			}
		}
		return flat_grads;
	}

	// Reconstructs the 1D gradient array received from the network back into 
	// the proper multi-dimensional tensor shapes and applies them to the local model.
	void InjectGradients(const std::vector<float>& global_flat_grads) {
		size_t offset = 0;

		auto primary_params = (config.type == TrainingConfig::DeviceType::CPU)
			? worker_cpu->GetModel()->parameters()
			: worker_gpu->GetModel()->parameters();

		for (size_t i = 0; i < primary_params.size(); ++i) {
			auto& p = primary_params[i];
			size_t numel = p.numel(); // Calculate numel() from the parameter itself, not the gradient!

			// If we skipped a turn (0 samples), the gradient tensor is undefined.
			// We must allocate a zero-filled tensor so we have memory to write the Master's incoming data!
			if (!p.grad().defined()) {
				p.mutable_grad() = torch::zeros_like(p);
			}

			// Temporary buffer in system RAM
			auto temp_tensor = torch::empty_like(p.grad().cpu().contiguous());
			std::memcpy(temp_tensor.data_ptr<float>(), global_flat_grads.data() + offset, numel * sizeof(float));

			// Write back to the main model (on its native device)
			if (p.device().is_cuda()) {
				torch::DeviceGuard device_guard(p.device());
				p.grad().copy_(temp_tensor.to(p.device()));
			}
			else {
				p.grad().copy_(temp_tensor.to(p.device()));
			}

			offset += numel;
		}
	}

public:
	void SetLogCallback(LogCallback callback) {
		log_callback = std::move(callback);
	}

	// Constructor for all available modes (CPU, GPU, Hybrid). 
	// It initializes the workers and threads based on the config.
	LocalTrainer(std::function<std::shared_ptr<IModel>()> model_builder, const TrainingConfig& config)
		: config(config), balancer(config.balancer)
	{
		if (config.random_seed.has_value()) {
			torch::manual_seed(config.random_seed.value());
			random_seed = true;
		}
		if (config.type == TrainingConfig::DeviceType::Hybrid) {
			std::cout << "[LocalTrainer] Hybrid mode" << std::endl;
			is_hybrid = true;
			model_gpu_ = model_builder();
			model_cpu_ = model_builder();
			if (torch::cuda::is_available()) {
				worker_gpu = std::make_unique<Worker>(model_gpu_, torch::kCUDA, config);
				std::cout << "[LocalTrainer] GPU initialized for Hybrid mode" << std::endl;
			}
			else {
				throw std::runtime_error("[LocalTrainer] CUDA not available, use CPU mode");
			}
			worker_cpu = std::make_unique<Worker>(model_cpu_, torch::kCPU, config);
			std::cout << "[LocalTrainer] CPU initialized for Hybrid mode" << std::endl;
			if (!random_seed) {
				std::cout << "[LocalTrainer] Warning: No random seed set. Starting to synchronize the models." << std::endl;
				SyncWeights(worker_gpu->GetModel(), worker_cpu->GetModel());
				std::cout << "[LocalTrainer] Done." << std::endl;
			}
			InitThreadPool();
			std::cout << "[LocalTrainer] Ready to work (Hybrid mode)." << std::endl;
		}
		else if (config.type == TrainingConfig::DeviceType::GPU) {
			if (torch::cuda::is_available()) {
				model_gpu_ = model_builder();
				worker_gpu = std::make_unique<Worker>(model_gpu_, torch::kCUDA, config);
				std::cout << "[LocalTrainer] GPU mode" << std::endl;
			}
			else {
				throw std::runtime_error("[LocalTrainer] CUDA not available, use CPU mode");
			}
			std::cout << "[LocalTrainer] Ready to work (GPU mode)." << std::endl;
		}
		else if (config.type == TrainingConfig::DeviceType::CPU) {
			model_cpu_ = model_builder();
			worker_cpu = std::make_unique<Worker>(model_cpu_, torch::kCPU, config);
			std::cout << "[LocalTrainer] CPU mode" << std::endl;
			std::cout << "[LocalTrainer] Ready to work (CPU mode)." << std::endl;
		}
		else {
			throw std::runtime_error("Unsupported device type");
		}
	}

	// Destructor: Safely shuts down background threads to prevent memory leaks or crashes
	~LocalTrainer() {
		if (thread_gpu.joinable() || thread_cpu.joinable()) {
			{
				std::lock_guard<std::mutex> lock(mtx);
				state = State::Stop;
				gpu_has_work = true; // Force wait condition to pass so threads can exit
				cpu_has_work = true;
			}
			cv_start.notify_all();
			if (thread_gpu.joinable()) thread_gpu.join();
			if (thread_cpu.joinable()) thread_cpu.join();
		}
	}

	// Main execution loop. Iterates over epochs and batches.
	template<typename DatasetType>
	void Fit(DatasetType user_data) {
		auto train_loop = [&](auto& data_loader) {
			for (int epoch = 1; epoch <= config.epochs; ++epoch) {
				int batch_index = 0;
				for (auto& raw_batch : *data_loader) {
					auto batch_start_time = std::chrono::high_resolution_clock::now();
					batch_index++;

					// 1. DYNAMIC DATASET SLICING (Each node takes its assigned chunk)
					bool has_data = true;
					torch::data::Example<> batch = { raw_batch.data, raw_batch.target };
					if (config.world_size > 1) {
						int total_size = raw_batch.data.size(0);
						int s_idx = std::min(net_start_idx, total_size);
						int e_idx = std::min(net_end_idx, total_size);
						if (s_idx < e_idx) {
							batch.data = raw_batch.data.slice(0, s_idx, e_idx);
							batch.target = raw_batch.target.slice(0, s_idx, e_idx);
						}
						else {
							has_data = false;
						}
					}

					// 2. HARDWARE PROFILER
					if (is_hybrid && !is_profiler) {
						auto profiler = DeviceProfiler::RunProfile(*worker_gpu, *worker_cpu, batch.data, batch.target);
						balancer.SetRatio(profiler.gpu_ratio);
						std::cout << "[LocalTrainer] Finished Profiler work" << std::endl;
						is_profiler = true;
					}

					float current_loss = 0.0f;
					std::function<void(double, int)> sync_hook = nullptr;

					// 3. NETWORK COMMUNICATION (Per batch)
					if (network_sync_callback_) {
						sync_hook = [this](double compute_time, int samples) {
							float current_ratio = is_hybrid ? balancer.GetRatio() : 1.0f;
							std::vector<float> my_local_grads = FlattenAndReduceLocalGradients(current_ratio);

							// Send compute time, receive new batch indices and global gradients
							auto result = network_sync_callback_(my_local_grads, compute_time, samples);

							InjectGradients(result.first);

							// UPDATE BOUNDARIES FOR THE NEXT BATCH!
							this->net_start_idx = result.second.start_index;
							this->net_end_idx = result.second.end_index;
							};
					}

					// 4. LOCAL TRAINING
					if (has_data) {
						// If we have data, proceed with standard training execution
						if (is_hybrid) current_loss = ExecuteHybridBatch(batch, sync_hook);
						else if (worker_gpu) current_loss = worker_gpu->TrainBatch(batch.data.to(torch::kCUDA, true), batch.target.to(torch::kCUDA, true), sync_hook);
						else if (worker_cpu) current_loss = worker_cpu->TrainBatch(batch.data, batch.target, sync_hook);
					}
					else {
						// If 0 samples assigned - we just check in with the network (sending 0 time, 0 samples)
						if (sync_hook) sync_hook(0.0, 0);

						// ==============================================================
						// A Worker WITHOUT data MUST still execute the optimizer step 
						// using the injected global gradients to remain identical to the Master!
						// ==============================================================
						if (is_hybrid) {
							worker_gpu->ApplyGrad();
							SyncWeights(worker_gpu->GetModel(), worker_cpu->GetModel());
						}
						else if (worker_gpu) worker_gpu->ApplyGrad();
						else if (worker_cpu) worker_cpu->ApplyGrad();
					}

					auto batch_end_time = std::chrono::high_resolution_clock::now();
					double batch_time = std::chrono::duration<double, std::milli>(batch_end_time - batch_start_time).count();

					// Update logging metrics
					if (epoch == 1 && batch_index == 1) ema_loss = current_loss;
					else ema_loss = 0.1f * current_loss + 0.9f * ema_loss;

					if (log_callback && (batch_index % config.log_interval_batches == 0)) {
						TrainingMetrics metrics;
						metrics.epoch = epoch;
						metrics.batch = batch_index;
						metrics.current_loss = current_loss;
						metrics.ema_loss = ema_loss;
						metrics.learning_rate = config.learning_rate;
						metrics.batch_time_ms = batch_time;
						metrics.local_gpu_ratio = balancer.GetRatio();
						metrics.is_gpu_only = balancer.IsGpuOnly();

						int processed_here = batch.data.size(0);
						if (batch_time > 0) {
							metrics.samples_per_second = static_cast<float>(processed_here / (batch_time / 1000.0));
						}

						metrics.worker_id = config.worker_id;
						metrics.world_size = config.world_size;
						metrics.local_chunk_size = processed_here;
						if (config.batch_size > 0) {
							metrics.cluster_load_share = static_cast<float>(processed_here) / config.batch_size;
						}
						log_callback(metrics);
					}
				}
			}
			};

		// Initialize DataLoader depending on user's shuffle configuration
		if (config.shuffle) {
			auto data_loader = DataFactory::CreateDataLoader<torch::data::samplers::RandomSampler>(user_data, config.batch_size, config.num_workers);
			train_loop(data_loader);
		}
		else {
			auto data_loader = DataFactory::CreateDataLoader<torch::data::samplers::SequentialSampler>(user_data, config.batch_size, config.num_workers);
			train_loop(data_loader);
		}
	}

	// Extracts the fully trained model and moves it back to system RAM
	std::shared_ptr<IModel> GetFinalModel() {
		if (worker_gpu) {
			auto final_model = worker_gpu->GetModel();
			final_model->to(torch::kCPU);
			return final_model;
		}
		else if (worker_cpu) {
			return worker_cpu->GetModel();
		}
		throw std::runtime_error("No active workers");
	}
};
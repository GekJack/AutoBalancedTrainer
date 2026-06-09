#pragma once
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include "Worker.hpp"

// A simple structure to hold the results of our hardware benchmark.
struct ProfileResult {
	float gpu_ratio; // The calculated ideal percentage of data for the GPU
	double time_gpu; // How long the GPU took to process the sample batch (in seconds)
	double time_cpu; // How long the CPU took to process the sample batch (in seconds)
};

// DeviceProfiler is a utility class used right before the main training loop starts.
// It runs a "test race" between the CPU and GPU using a small sample batch 
// to give the LoadBalancer a perfect starting ratio, avoiding initial lag.
class DeviceProfiler {
public:
	// Runs a benchmark on both devices and calculates their relative performance.
	static ProfileResult RunProfile(Worker& worker_gpu, Worker& worker_cpu, torch::Tensor sample_data, torch::Tensor sample_target) {
		std::cout << "Profiling GPU..." << std::endl;

		// Move sample data to GPU memory (VRAM)
		auto gpu_data = sample_data.to(torch::kCUDA);
		auto gpu_target = sample_target.to(torch::kCUDA);

		// WARM-UP PASS: 
		// The first time a CUDA kernel runs, there is a massive overhead for memory 
		// allocation and context initialization. We run a "dummy" batch first and 
		// ignore its time so it doesn't ruin our benchmark results.
		worker_gpu.TrainBatch(gpu_data, gpu_target);

		// synchronize() forces the CPU to stop and wait until the GPU physically 
		// finishes the warm-up before we start the actual timer.
		torch::cuda::synchronize();

		// ACTUAL GPU PROFILING
		auto start_gpu = std::chrono::high_resolution_clock::now();
		worker_gpu.TrainBatch(gpu_data, gpu_target);

		// GPU operations are asynchronous! If we don't call synchronize() here, 
		// the timer will stop immediately while the GPU is still working in the background.
		torch::cuda::synchronize();
		auto end_gpu = std::chrono::high_resolution_clock::now();

		// Move sample data to CPU memory (RAM)
		auto cpu_data = sample_data.to(torch::kCPU);
		auto cpu_target = sample_target.to(torch::kCPU);

		std::cout << "Profiling CPU..." << std::endl;

		// ACTUAL CPU PROFILING
		// CPU operations are synchronous by default in this context, so we don't 
		// strictly need a warm-up or a synchronize() call here.
		auto start_cpu = std::chrono::high_resolution_clock::now();
		worker_cpu.TrainBatch(cpu_data, cpu_target);
		auto end_cpu = std::chrono::high_resolution_clock::now();

		// Calculate exact duration in seconds
		double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
		double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();

		// Calculate the ideal starting ratio.
		// Math logic: If CPU takes 9 seconds and GPU takes 1 second, the total time is 10s.
		// CPU time (9) / Total time (10) = 0.9. So, the GPU should get 90% of the workload.
		float ratio = static_cast<float>(time_cpu / (time_cpu + time_gpu));

		// Print the benchmark results to the console
		std::cout << "GPU Time: " << time_gpu << " seconds" << std::endl;
		std::cout << "CPU Time: " << time_cpu << " seconds" << std::endl;
		std::cout << "GPU Ratio: " << ratio * 100.0f << " %" << std::endl;
		std::cout << "CPU Ratio: " << (1.0f - ratio) * 100.0f << " %" << std::endl;

		return{ ratio, time_gpu, time_cpu };
	}
};
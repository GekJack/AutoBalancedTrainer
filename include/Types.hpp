#pragma once
#include <torch/torch.h>
#include <optional>

// Hyperparameters structure used for network transmission.
// Primitive types are used to ensure safe memory copying (serialization).
struct NetworkHyperparams {
	float learning_rate;
	int epochs;
	int batch_size;
	bool shuffle;
	int optimizer_type; // Passed as int for easy network serialization
	float weight_decay;
	float momentum;
	int loss_type;      // Passed as int for easy network serialization

	// Safe flags for std::optional values handling over the network
	bool has_seed;
	int random_seed;

	bool has_clip;
	float clip_grad_norm;
};

// 1. Initialization payload (The "Passport")
// Sent from Master to Workers during the initial handshake to synchronize settings.
struct InitPayload {
	int worker_id;             // Assigned ID for the connecting worker
	int world_size;            // Total number of nodes in the cluster
	int initial_start_idx;     // Starting index of the dataset for the first batch
	int initial_end_idx;       // Ending index of the dataset for the first batch
	NetworkHyperparams hyperparams; // Training settings to enforce consistency
};

// 2. Package for global data distribution
// Sent by Master via PUB/SUB BEFORE each epoch to assign data slices.
struct EpochAssignment {
	int start_index;
	int end_index;
	float current_ratio; // Node's assigned quota (e.g., 0.5 means 50% of the global data)
};

// 3. Report from the Worker
// Sent back to Master via REQ/REP after processing a batch. Used for dynamic ratio updates.
struct WorkerReport {
	int worker_id;             // Identifier of the reporting worker
	float compute_time_ms;     // Time taken to process the assigned chunk
	int processed_samples;     // Actual number of images/samples processed
};

// Base interface for all neural network models in this system.
// By inheriting from torch::nn::Module and defining a pure virtual forward() method,
// we ensure that ANY custom model can be plugged into our LocalTrainer seamlessly.
struct IModel : public torch::nn::Module {
	virtual ~IModel() = default;

	// The core feed-forward logic of the neural network
	virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// Configuration settings specifically for the dynamic LoadBalancer.
// These parameters control how aggressively the system shifts the workload 
// between the CPU and GPU to avoid the "straggler effect".
struct BalancerConfig {
	float max_gpu_ratio = 0.98f;         // Max allowed load for GPU (leaves 2% to keep CPU active)
	float min_gpu_ratio = 0.20f;         // Min allowed load for GPU (prevents CPU from being overloaded)
	int test_interval = 50;              // Batches to wait before giving a "killed" CPU a second chance
	float cpu_slowdown_factor = 1.5f;    // Kill threshold: CPU takes 1.5x longer than GPU
	double min_time_diff_ms_adjust = 0.5; // Micro-tuning: Minimum ms difference to trigger a ratio shift
	double min_time_diff_ms_kill = 30.0;  // Kill switch: Absolute ms delay to drop the CPU entirely
	float ratio_step = 0.01f;            // How much to adjust the ratio per batch (1% step)
	float test_start_ratio = 0.85f;      // Default starting state (85% of data to GPU, 15% to CPU)
};

// Structure to hold all real-time metrics during training.
// Used by the Callback system to stream data to the user interface.
struct TrainingMetrics {
	// --- 1. Machine Learning (ML) Metrics ---
	int epoch = 0;                  // Current training epoch
	int batch = 0;                  // Current batch index
	int total_batches = 0;          // Total number of batches in the current epoch
	float current_loss = 0.0f;      // Raw loss of the current batch
	float ema_loss = 0.0f;          // Exponential moving average loss (smoothed trend)
	float learning_rate = 0.001f;   // Current active learning rate

	// --- 2. Local Hardware Performance Metrics ---
	double batch_time_ms = 0.0;     // Total time taken for the batch in milliseconds
	float local_gpu_ratio = 1.0f;   // Percentage of workload sent to GPU inside this machine
	bool is_gpu_only = false;       // Flag indicating if CPU was dropped due to bottleneck
	float samples_per_second = 0.f; // Processing speed (throughput)

	// --- 3. Global Network Metrics (Cluster Topology) ---
	int worker_id = 0;              // Node identifier: 0 = Master, 1+ = Workers
	int world_size = 1;             // Total number of machines in the cluster
	int local_chunk_size = 0;       // Number of samples assigned to this machine by the network
	float cluster_load_share = 0.f; // Fraction of the global batch handled by this machine (0.0 - 1.0)
};

// Comprehensive configuration for the training session.
// It bundles standard machine learning hyperparameters, optimizer settings, 
// and the balancer configuration into a single, clean object.
struct TrainingConfig {
	// Standard Machine Learning Hyperparameters
	float learning_rate = 0.001f; // The step size at each iteration while moving toward a minimum
	int epochs = 100;             // How many times to loop over the entire dataset
	int batch_size = 64;          // Total number of samples processed before the model is updated
	bool shuffle = false;         // Whether to randomize data order (critical for good training)
	int num_workers = 2;          // Number of background OS threads for the DataLoader

	enum class DeviceType { CPU, GPU, Hybrid };
	DeviceType type = DeviceType::CPU;

	// Optimizer selection
	enum class OptimizerType {
		Adam,
		SGD,
		RMSProp
	};
	OptimizerType optimizer_type = OptimizerType::Adam;

	// Cluster and logical distribution variables
	int world_size = 1;           // Total nodes in the cluster
	int worker_id = 0;            // Current node identifier
	int net_start_idx = 0;        // Start index for the network batch slice
	int net_end_idx = 0;          // End index for the network batch slice

	// Advanced optimizer settings
	float weight_decay = 0.0f;    // L2 regularization penalty (prevents overfitting)
	float momentum = 0.0f;        // Accelerates gradients in the right direction (mainly for SGD/RMSProp)

	// Mathematical function to calculate the error between predictions and targets
	enum class LossType { MSE, CrossEntropy, L1 };
	LossType loss_type = LossType::MSE;
	int log_interval_batches = 10; // Frequency of logging updates

	// std::optional is used here because these settings are not always required.
	// nullopt means the feature is turned off by default.
	std::optional<int> random_seed = std::nullopt;       // For reproducible experiments and network sync
	std::optional<float> clip_grad_norm = std::nullopt;  // Prevents the "exploding gradient" problem

	// Nested configuration for our custom dynamic hardware balancing
	BalancerConfig balancer;

	// Overwrites local training settings with the ones received from the Master node
	void MergeFromMaster(const NetworkHyperparams& master_params) {
		learning_rate = master_params.learning_rate;
		epochs = master_params.epochs;
		batch_size = master_params.batch_size;
		shuffle = master_params.shuffle;

		optimizer_type = static_cast<OptimizerType>(master_params.optimizer_type);
		weight_decay = master_params.weight_decay;
		momentum = master_params.momentum;
		loss_type = static_cast<LossType>(master_params.loss_type);

		if (master_params.has_seed) {
			random_seed = master_params.random_seed;
		}
		else {
			random_seed = std::nullopt;
		}

		if (master_params.has_clip) {
			clip_grad_norm = master_params.clip_grad_norm;
		}
		else {
			clip_grad_norm = std::nullopt;
		}
	}
};

// Configuration for setting up the ZeroMQ network topology
struct NetworkConfig {
	enum class Role { Master, Worker };

	Role role = Role::Worker;
	std::string master_ip = "127.0.0.1";

	int port_req = 5555; // Port for initial handshake and gradient reports (REQ/REP)
	int port_pub = 5556; // Port for barrier commands and global weights broadcast (PUB/SUB)

	// Priority hardware mode specifically for this physical machine
	TrainingConfig::DeviceType local_device_type = TrainingConfig::DeviceType::CPU;

	// How many workers the Master should wait for before starting the training loop (Master only)
	int expected_workers = 1;

	// Unique identifier for each node (0 for Master, 1..N for Workers)
	int worker_id = 0;

	// Total number of machines in the cluster (Master + Workers)
	int world_size = 1;
};
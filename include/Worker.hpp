#pragma once
#include <memory>
#include <torch/torch.h>
#include "Types.hpp"
#include <functional>

// Worker class represents an isolated computing node (either CPU or GPU).
// It holds its own copy of the model, optimizer, and loss function.
// The core design choice here is separating the gradient calculation from the weight update,
// which is strictly necessary for our custom distributed training architecture.
class Worker {
private:
	std::shared_ptr<IModel> model; // Pointer to the neural network
	torch::Device device;          // The hardware this worker runs on (CPU or CUDA)
	std::unique_ptr<torch::optim::Optimizer> optimizer; // Algorithm that updates the weights (Adam, SGD, etc.)
	TrainingConfig config;         // Hyperparameters like learning rate
	std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_function; // Math formula to calculate the error

public:
	// Constructor sets up the worker with the given model, device, and settings
	Worker(std::shared_ptr<IModel> model, torch::Device dev, TrainingConfig opts) : model(model), device(dev),
		config(opts)
	{
		// Move the model to the target hardware's memory (RAM for CPU, VRAM for GPU)
		model->to(device);

		// Initialize the specific optimizer based on our config
		switch (config.optimizer_type) {
		case TrainingConfig::OptimizerType::Adam:
			optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));
			break;
		case TrainingConfig::OptimizerType::SGD:
			optimizer = std::make_unique<torch::optim::SGD>(model->parameters(), torch::optim::SGDOptions(config.learning_rate).momentum(config.momentum).weight_decay(config.weight_decay));
			break;
		case TrainingConfig::OptimizerType::RMSProp:
			optimizer = std::make_unique<torch::optim::RMSprop>(model->parameters(), torch::optim::RMSpropOptions(config.learning_rate).momentum(config.momentum).weight_decay(config.weight_decay));
			break;
		}

		// Bind the selected mathematical loss function
		switch (config.loss_type) {
		case TrainingConfig::LossType::MSE:
			loss_function = [](torch::Tensor pred, torch::Tensor target) {
				return torch::mse_loss(pred, target);
				};
			break;
		case TrainingConfig::LossType::CrossEntropy:
			loss_function = [](torch::Tensor pred, torch::Tensor target) {
				return torch::cross_entropy_loss(pred, target);
				};
			break;
		case TrainingConfig::LossType::L1:
			loss_function = [](torch::Tensor pred, torch::Tensor target) {
				return torch::l1_loss(pred, target);
				};
			break;
		}
	}

	// Standard training loop step for single-device training.
	// It computes the gradients and immediately applies them.
	float TrainBatch(torch::Tensor data, torch::Tensor target, std::function<void(double, int)> sync_hook = nullptr) {
		auto start = std::chrono::high_resolution_clock::now();
		float loss_value = ComputeGrad(data, target);
		auto end = std::chrono::high_resolution_clock::now();
		double compute_time = std::chrono::duration<double, std::milli>(end - start).count();
		int processed_samples = data.size(0);
		if (sync_hook) {
			sync_hook(compute_time, processed_samples);
		}
		ApplyGrad();
		return loss_value;
	}

	// Calculates gradients WITHOUT updating the model's weights.
	// This allows the Master node to collect and average gradients from multiple 
	// workers before actually taking a step.
	float ComputeGrad(torch::Tensor data, torch::Tensor target) {
		// Enable training mode (important for things like Dropout or BatchNorm)
		model->train();

		// Clear old gradients from the previous batch to prevent them from summing up
		optimizer->zero_grad();

		// Forward pass: get the model's predictions for this batch
		torch::Tensor prediction = model->forward(data);

		// Calculate the error
		torch::Tensor loss = loss_function(prediction, target);

		// Backward pass: calculate mathematical gradients for all weights
		loss.backward();

		return loss.item<float>();
	}

	// Updates the model's weights using the gradients we just calculated.
	// In a hybrid setup, this is called AFTER the Master averages gradients from all GPUs/CPUs.
	void ApplyGrad() {
		// Optional: clip gradients to a maximum value to prevent the "exploding gradient" problem
		if (config.clip_grad_norm.has_value()) {
			torch::nn::utils::clip_grad_norm_(model->parameters(), config.clip_grad_norm.value());
		}

		// Take a step to adjust the weights
		optimizer->step();
	}

	// Returns the model pointer, mainly used by the Master to sync weights between workers
	std::shared_ptr<IModel> GetModel() const {
		return model;
	}
};
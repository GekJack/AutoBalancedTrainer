#include <torch/torch.h>
#include <iostream>
#include "include/NetworkTrainer.hpp"

// ==============================================================================
// 1. DEFINE THE NEURAL NETWORK
// We create a standard Multi-Layer Perceptron (MLP) or Convolutional Network. 
// By inheriting from our custom 'IModel' interface, we guarantee it can be 
// seamlessly plugged into our LocalTrainer/NetworkTrainer architecture.
// ==============================================================================
struct Net : public IModel {
    Net() {
        // TODO: Register your neural network layers here.
        // The register_module function is required for the framework to track the weights.
        // Example: fc1 = register_module("fc1", torch::nn::Linear(2, 1028));
    }

    torch::Tensor forward(torch::Tensor x) override {
        // TODO: Define the forward pass logic here.
        // Example: x = torch::relu(fc1->forward(x));
        //          return fc2->forward(x);
        return x; 
    }

    // TODO: Declare your layer variables here.
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

// ==============================================================================
// 2. DEFINE THE DATASET
// A custom dataset generator or loader.
// Inherits from torch::data::Dataset to be compatible with PyTorch DataLoaders.
// ==============================================================================
class MyDataSet : public torch::data::Dataset<MyDataSet> {
private:
    // TODO: Add variables to store your data 
    // (e.g., inputs_ and targets_ tensors).

public:
    MyDataSet() {
        // TODO: Initialize or load data (e.g., from .pt or .bin files).
    }
    
    torch::data::Example<> get(size_t index) override {
        // TODO: Return a single data sample and its label by index.
        // Example: return { inputs_[index], targets_[index] };
        
        // Dummy return to ensure the file compiles without errors:
        return { torch::empty({1}), torch::empty({1}) }; 
    }

    torch::optional<size_t> size() const override {
        // TODO: Return the total number of samples in the dataset.
        // Example: return inputs_.size(0);
        return 0; 
    }
};

// ==============================================================================
// 3. ENTRY POINT (Framework Initialization and Execution)
// ==============================================================================
int main() {
    try {
        // MODEL BUILDER (FACTORY)
        // A lambda function that allows the framework to create isolated 
        // instances of your model for each computational thread/node.
        auto model_builder = []() {
            return std::make_shared<Net>();
        };

        // DATASET INITIALIZATION
        auto my_dataset = MyDataSet();

        // ---------------------------------------------------------
        // TODO: NETWORK CONFIGURATION (ZeroMQ)
        // Uncomment and configure this block to run in a cluster:
        // ---------------------------------------------------------
        /*
        NetworkConfig net_config;
        net_config.role = NetworkConfig::Role::Master; // Or NetworkConfig::Role::Worker
        net_config.expected_workers = 1;
        net_config.master_ip = "127.0.0.1"; // Specify Master IP if this is a Worker
        net_config.port_req = 5555;
        net_config.port_pub = 5556;
        net_config.local_device_type = TrainingConfig::DeviceType::GPU;
        */

        // ---------------------------------------------------------
        // TODO: TRAINING HYPERPARAMETERS CONFIGURATION
        // Uncomment and set the required parameters:
        // ---------------------------------------------------------
        /*
        TrainingConfig train_config;
        train_config.epochs = 50;
        train_config.batch_size = 4096; // For clusters, it is better to set large values
        train_config.optimizer_type = TrainingConfig::OptimizerType::Adam;
        train_config.loss_type = TrainingConfig::LossType::CrossEntropy;
        */

        // ---------------------------------------------------------
        // TODO: STARTING THE FRAMEWORK
        // Create a NetworkTrainer instance and pass the configuration:
        // ---------------------------------------------------------
        /*
        std::cout << "[INFO] Initializing AutoBalancedTrainer..." << std::endl;
        NetworkTrainer orchestrator(train_config, net_config);
        
        // Optional: Configure console log output
        orchestrator.SetLogCallback([](const TrainingMetrics& m) {
            std::cout << "Epoch: " << m.epoch << " | Batch: " << m.batch 
                      << " | Loss: " << m.ema_loss << std::endl;
        });

        // Start distributed training
        orchestrator.Start(model_builder, std::move(my_dataset));
        */
        
        std::cout << "[SUCCESS] Template built successfully. Fill in the TODO blocks to start training." << std::endl;

    }
    catch (const c10::Error& e) {
        // Catch block for PyTorch/LibTorch internal errors (e.g., CUDA out of memory)
        std::cerr << "\n[PYTORCH FATAL ERROR]: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        // Catch block for Framework Configuration errors or Threading errors
        std::cerr << "\n[FRAMEWORK ERROR]: " << e.what() << std::endl;
    }

    return 0;
}

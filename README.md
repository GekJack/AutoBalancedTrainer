# 🚀 AutoBalancedTrainer SDK

**AutoBalancedTrainer** — це легкий Header-Only C++ фреймворк для гібридного (CPU+GPU) та розподіленого навчання нейронних мереж у локальній мережі. Він дозволяє об'єднувати обчислювальні вузли різної потужності, автоматично балансуючи навантаження між ними за допомогою ZeroMQ та LibTorch.

## 🛠 Системні вимоги

* **ОС:** Windows 10/11 (64-bit).
* **Інструментарій:** Visual Studio 2019/2022 (MSVC, підтримка C++17).
* **Система збирання:** CMake 3.18+.
* **Апаратне прискорення:** NVIDIA CUDA Toolkit 12.1+ (Опціонально).

## 📦 Встановлення

### 1. LibTorch (PyTorch C++ API)
Завантажте попередньо скомпільовану версію LibTorch (Release) з [офіційного сайту PyTorch](https://pytorch.org/get-started/locally/). Оберіть версію **CUDA 12.1** (або новішу) для підтримки GPU. Розпакуйте архів у `C:\libtorch`.

### 2. vcpkg (Менеджер пакетів для ZeroMQ)
Фреймворк використовує `vcpkg` для автоматичного завантаження `cppzmq`. Відкрийте `cmd` і виконайте:
```cmd
cd C:\
git clone [https://github.com/microsoft/vcpkg.git](https://github.com/microsoft/vcpkg.git)
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
```
### 3. 🚀 Швидкий старт
Клонуйте цей репозиторій.

Відкрийте папку у Visual Studio (через "Open Folder").

CMake автоматично знайде файл vcpkg.json і завантажить ZeroMQ.

У файлі CMakeLists.txt перевірте правильність шляху до LibTorch (за замовчуванням C:/libtorch).

Зберіть проєкт та запустіть main_template.cpp.

У шаблоні main_template.cpp реалізовано приклад створення мережі та дата-сету,а також конфігурація координатора (Master) кластера.

### Частина 4. Файл `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(AutoBalancedTrainer VERSION 1.0 LANGUAGES CXX)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
add_executable(${PROJECT_NAME} main_template.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} cppzmq)
set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
```
##Частина 5.main_template.cpp
```c++
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
```

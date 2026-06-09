#pragma once
#include <torch/torch.h>

// A utility namespace designed to simplify the creation of PyTorch DataLoaders.
// A DataLoader is a critical component that feeds data to the neural network in batches,
// handles shuffling, and loads data in parallel using background CPU threads.
namespace DataFactory {

	// Template function to create a DataLoader for any custom dataset and sampling strategy.
	// - SamplerType: Dictates how data is picked (e.g., RandomSampler for training, Sequential for testing).
	// - DatasetType: The specific user-defined dataset class being passed.
	template<typename SamplerType, typename DatasetType>
	auto CreateDataLoader(DatasetType& dataset, int batch_size, int workers) {

		// 1. Map the dataset with a Stack transform.
		// Why is this necessary? A raw dataset returns individual data items (e.g., 1 image).
		// However, neural networks train on "batches" of data simultaneously.
		// The Stack<>() transform automatically takes 'batch_size' individual items 
		// and groups (stacks) them together into a single large tensor (a batch).
		auto dataset_mapped = std::move(dataset).map(torch::data::transforms::Stack<>());

		// 2. Configure the DataLoader options.
		// - batch_size: How many individual samples to group together per iteration.
		// - workers: How many background OS threads to dedicate strictly to loading 
		//   and preprocessing data from RAM/Disk, preventing the GPU from starving.
		auto options = torch::data::DataLoaderOptions().batch_size(batch_size).workers(workers);

		// 3. Create and return the actual DataLoader object.
		// We use std::move to transfer ownership of the dataset into the DataLoader memory space.
		return torch::data::make_data_loader<SamplerType>(std::move(dataset_mapped), options);
	}
}
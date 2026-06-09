#pragma once 
#include <atomic>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "Types.hpp"

// LoadBalancer acts as the "brain" of the hybrid training system.
// It dynamically adjusts how much data goes to the GPU versus the CPU
// based on their real-time performance (execution time).
class LoadBalancer {
private:
	// atomic ensures thread safety since Master and Workers might access this concurrently
	std::atomic<float> current_gpu_ratio;

	// Flag to completely disable the CPU if it becomes a severe bottleneck
	bool is_gpu_only = false;

	// Counter used to periodically check if we should try using the CPU again
	int test_counter = 0;

	BalancerConfig config; // Holds thresholds, steps, and limits

public:
	// Initializes the balancer with a starting ratio (e.g., 0.85 for 85% GPU)
	LoadBalancer(BalancerConfig cfg) : config(cfg), current_gpu_ratio(cfg.test_start_ratio) {}

	// Returns the current percentage of data that should be sent to the GPU
	float GetRatio() const {
		return current_gpu_ratio.load();
	}

	// Checks if the CPU has been dropped from the training process
	bool IsGpuOnly() const {
		return is_gpu_only;
	}

	// Periodically returns 'true' to tell the Master node: 
	// "Hey, let's give the CPU a small test batch to see if it recovered."
	bool ShouldTestCPU() {
		if (!is_gpu_only) return false;

		if (++test_counter >= config.test_interval) {
			test_counter = 0;
			return true;
		}
		return false;
	}

	// Manually override the current ratio
	void SetRatio(float new_ratio) {
		current_gpu_ratio.store(new_ratio);
	}

	// Manually force the system back into hybrid (CPU + GPU) mode
	void ForceHybrid() {
		is_gpu_only = false;
		current_gpu_ratio.store(config.test_start_ratio);
	}

	// The core algorithm: adjusts the load based on the time it took to process the last batch
	void Update(double time_gpu, double time_cpu) {
		if (is_gpu_only) return; // If CPU is already turned off, do nothing

		double time_diff = std::abs(time_cpu - time_gpu);
		float ratio = current_gpu_ratio.load();

		// 1. MICRO-TUNING ZONE: 
		// If the time difference is noticeable, shift a small % of data to the faster device.
		if (time_diff > config.min_time_diff_ms_adjust) {
			if (time_gpu < time_cpu) {
				ratio += config.ratio_step; // GPU is faster, give it more work
			}
			else {
				ratio -= config.ratio_step; // CPU is faster (rare, but possible), give it more work
			}
		}

		// 2. KILL SWITCH (Bottleneck detection):
		// If the CPU is significantly slower (e.g., 3x slower) AND the absolute delay is huge.
		bool is_cpu_bottleneck = (time_cpu > time_gpu * config.cpu_slowdown_factor) &&
			((time_cpu - time_gpu) > config.min_time_diff_ms_kill);

		// If CPU is holding everything back, or GPU is already doing ~99% of the work, drop the CPU.
		if (ratio >= config.max_gpu_ratio || is_cpu_bottleneck) {
			std::cout << "[Balancer] CPU is slow. Turn on GPU only." << std::endl;
			is_gpu_only = true;
			test_counter = 0; // Reset counter so we don't immediately test the CPU again
		}

		// Clamp the ratio to prevent logical errors (e.g., ratio going above 1.0 or below 0.0)
		ratio = std::max(config.min_gpu_ratio, std::min(config.max_gpu_ratio, ratio));

		current_gpu_ratio.store(ratio);
	}
};
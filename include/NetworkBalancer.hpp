#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "Types.hpp"

// NetworkBalancer handles the global (cluster-level) load balancing.
// It dynamically calculates how much of the dataset each physical machine 
// should process based on its real-time throughput (samples per second).
class NetworkBalancer {
private:
    int total_nodes_;
    std::vector<float> ratios_;

    // Minimum 5% data quota for any node.
    // This safeguard prevents a temporarily slow node from dropping to 0% 
    // and never getting another chance to prove its speed.
    const float MIN_RATIO = 0.05f;

public:
    // Initializes the balancer, dividing the initial workload equally among all nodes.
    NetworkBalancer(int nodes) : total_nodes_(nodes) {
        float initial_ratio = 1.0f / total_nodes_;
        ratios_.resize(total_nodes_, initial_ratio);
        std::cout << "[NetBalancer] Initialized. Nodes: " << total_nodes_ << "\n";
    }

    // Slices the current global batch into logical chunks based on the calculated ratios.
    // Returns a vector of EpochAssignments (start and end indices) for each node.
    std::vector<EpochAssignment> GenerateAssignments(int current_batch_size) {
        std::vector<EpochAssignment> assignments(total_nodes_);
        int current_start = 0;
        int remaining_samples = current_batch_size;

        for (int i = 0; i < total_nodes_; ++i) {
            int chunk_size = 0;

            if (i == total_nodes_ - 1) {
                // The last node takes ALL remaining samples.
                // This perfectly handles rounding errors and undersized tail batches.
                chunk_size = remaining_samples;
            }
            else {
                // Calculate chunk size based on ratio, guaranteeing at least 1 sample 
                // if there is enough remaining data to distribute.
                chunk_size = static_cast<int>(current_batch_size * ratios_[i]);
                if (remaining_samples > total_nodes_ - i) {
                    chunk_size = std::max(1, chunk_size);
                }
                chunk_size = std::min(chunk_size, remaining_samples);
            }

            assignments[i].start_index = current_start;
            assignments[i].end_index = current_start + chunk_size;
            assignments[i].current_ratio = ratios_[i];

            current_start += chunk_size;
            remaining_samples -= chunk_size;
        }
        return assignments;
    }

    // Recalculates the distribution ratios based on the telemetry (WorkerReports)
    // received from all nodes after a batch completes.
    void UpdateRatios(const std::vector<WorkerReport>& reports) {
        double total_speed = 0.0;
        std::vector<double> speeds(total_nodes_, 0.0);

        // 1. Calculate the absolute processing speed (throughput) for each node
        for (const auto& rep : reports) {
            // Speed = processed samples / elapsed time. Added 1e-5 to prevent division by zero.
            double speed = rep.processed_samples / (rep.compute_time_ms + 1e-5);
            speeds[rep.worker_id] = speed;
            total_speed += speed;
        }

        // 2. Calculate new logical quotas based on relative speeds
        float total_ratio = 0.0f;
        for (int i = 0; i < total_nodes_; ++i) {
            float calculated_ratio = static_cast<float>(speeds[i] / total_speed);

            // Safeguard: Clamp the ratio to MIN_RATIO so no node is permanently starved
            ratios_[i] = std::max(MIN_RATIO, calculated_ratio);
            total_ratio += ratios_[i];
        }

        // 3. Normalize ratios so they sum up exactly to 1.0 (100%)
        // Because of the MIN_RATIO clamp, the sum might exceed 1.0 before this step.
        for (int i = 0; i < total_nodes_; ++i) {
            ratios_[i] /= total_ratio;
        }
    }
};
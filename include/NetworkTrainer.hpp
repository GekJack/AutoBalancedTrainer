#pragma once
#include <torch/torch.h>
#include <zmq.hpp>
#include <memory>
#include <functional>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include "LocalTrainer.hpp"
#include "Types.hpp"
#include "NetworkBalancer.hpp"

// NetworkTrainer is the top-level orchestration class for the cluster.
// It sets up the ZeroMQ sockets, handles the initial handshake (parameter syncing),
// and manages the gradient aggregation and barrier synchronization between nodes.
class NetworkTrainer {
public:
    using LogCallback = std::function<void(const TrainingMetrics&)>;
private:
    NetworkConfig net_config_;
    TrainingConfig train_config_;
    std::unique_ptr<LocalTrainer> local_engine_ = nullptr;

    zmq::context_t m_context;
    zmq::socket_t m_req_rep;
    zmq::socket_t m_pub_sub;
    LogCallback net_log_callback = nullptr; // Function pointer for logging training progress and metrics

public:
    NetworkTrainer(TrainingConfig& train_cfg, const NetworkConfig& net_cfg)
        : train_config_(train_cfg), net_config_(net_cfg), m_context(1)
    {
        train_config_.type = net_config_.local_device_type;
        int64_t exchanged_seed = 0;

        net_config_.worker_id = (net_config_.role == NetworkConfig::Role::Master) ? 0 : net_config_.worker_id;
        net_config_.world_size = (net_config_.role == NetworkConfig::Role::Master) ? (net_config_.expected_workers + 1) : net_config_.world_size;

        if (net_config_.role == NetworkConfig::Role::Master) {
            // ==========================================================
            // MASTER NODE: Calculate initial data chunks for all nodes
            // ==========================================================
            NetworkBalancer init_balancer(net_config_.world_size);
            std::vector<EpochAssignment> initial_assignments = init_balancer.GenerateAssignments(train_config_.batch_size);

            // Master assigns the 0-th data chunk to itself
            train_config_.net_start_idx = initial_assignments[0].start_index;
            train_config_.net_end_idx = initial_assignments[0].end_index;
            train_config_.world_size = net_config_.world_size;
            train_config_.worker_id = net_config_.worker_id;

            // Seed generation for global RNG synchronization
            if (!train_config_.random_seed.has_value()) {
                std::random_device rd; exchanged_seed = rd();
                train_config_.random_seed = exchanged_seed;
            }
            else { exchanged_seed = train_config_.random_seed.value(); }

            // Initialize sockets for communication
            m_req_rep = zmq::socket_t(m_context, zmq::socket_type::rep);
            m_req_rep.bind("tcp://*:" + std::to_string(net_config_.port_req));
            m_pub_sub = zmq::socket_t(m_context, zmq::socket_type::pub);
            m_pub_sub.bind("tcp://*:" + std::to_string(net_config_.port_pub));

            // Fault Tolerance: Set network timeout to prevent infinite deadlocks
            int timeout_ms = 15000;
            m_req_rep.set(zmq::sockopt::rcvtimeo, timeout_ms);
            m_pub_sub.set(zmq::sockopt::rcvtimeo, timeout_ms);

            // DISTRIBUTE INITIALIZATION PAYLOADS ("PASSPORTS") TO WORKERS
            for (int i = 0; i < net_config_.expected_workers; ++i) {
                zmq::message_t req;
                m_req_rep.recv(req, zmq::recv_flags::none);

                InitPayload payload = {};
                payload.worker_id = i + 1;
                payload.world_size = net_config_.world_size;

                // ASSIGN THE SPECIFIC DATA SLICE TO THIS WORKER
                payload.initial_start_idx = initial_assignments[i + 1].start_index;
                payload.initial_end_idx = initial_assignments[i + 1].end_index;

                // Broadcast Master's hyperparameters to ensure configuration consistency
                payload.hyperparams.learning_rate = train_config_.learning_rate;
                payload.hyperparams.epochs = train_config_.epochs;
                payload.hyperparams.batch_size = train_config_.batch_size;
                payload.hyperparams.shuffle = train_config_.shuffle;
                payload.hyperparams.optimizer_type = static_cast<int>(train_config_.optimizer_type);
                payload.hyperparams.weight_decay = train_config_.weight_decay;
                payload.hyperparams.momentum = train_config_.momentum;
                payload.hyperparams.loss_type = static_cast<int>(train_config_.loss_type);
                payload.hyperparams.has_seed = train_config_.random_seed.has_value();

                if (payload.hyperparams.has_seed) {
                    payload.hyperparams.random_seed = static_cast<int>(train_config_.random_seed.value());
                }

                payload.hyperparams.has_clip = train_config_.clip_grad_norm.has_value();
                if (payload.hyperparams.has_clip) {
                    payload.hyperparams.clip_grad_norm = train_config_.clip_grad_norm.value();
                }

                // Send the payload back to the connecting Worker
                zmq::message_t rep(sizeof(InitPayload));
                std::memcpy(rep.data(), &payload, sizeof(InitPayload));
                m_req_rep.send(rep, zmq::send_flags::none);
            }
        }
        else {
            // ==========================================================
            // WORKER NODE: Connect to Master and request initial setup
            // ==========================================================
            m_req_rep = zmq::socket_t(m_context, zmq::socket_type::req);
            m_req_rep.connect("tcp://" + net_config_.master_ip + ":" + std::to_string(net_config_.port_req));
            m_pub_sub = zmq::socket_t(m_context, zmq::socket_type::sub);
            m_pub_sub.connect("tcp://" + net_config_.master_ip + ":" + std::to_string(net_config_.port_pub));
            m_pub_sub.set(zmq::sockopt::subscribe, "");

            // Perform handshake
            std::string msg = "HELLO_MASTER";
            zmq::message_t req(msg.size());
            std::memcpy(req.data(), msg.data(), msg.size());
            m_req_rep.send(req, zmq::send_flags::none);

            zmq::message_t rep;
            m_req_rep.recv(rep, zmq::recv_flags::none);

            InitPayload payload;
            std::memcpy(&payload, rep.data(), sizeof(InitPayload));

            net_config_.worker_id = payload.worker_id;
            net_config_.world_size = payload.world_size;

            // WORKER RECORDS ITS ASSIGNED DATA SLICE AND SETTINGS
            train_config_.world_size = payload.world_size;
            train_config_.worker_id = payload.worker_id;
            train_config_.net_start_idx = payload.initial_start_idx;
            train_config_.net_end_idx = payload.initial_end_idx;

            train_config_.MergeFromMaster(payload.hyperparams);
            exchanged_seed = train_config_.random_seed.value();

            std::cout << "\n==================================================\n";
            std::cout << "[WORKER INIT] Successfully connected to Master!\n";
            std::cout << " -> My Identity  : WORKER-" << net_config_.worker_id
                << " (out of " << net_config_.world_size << " total nodes)\n";
            std::cout << " -> Init Slice   : [" << train_config_.net_start_idx
                << " ... " << train_config_.net_end_idx << "]\n";
            std::cout << " -> Random Seed  : " << exchanged_seed << "\n";
            std::cout << " -> Batch Size   : " << train_config_.batch_size << "\n";
            std::cout << " -> Learning Rate: " << train_config_.learning_rate << "\n";
            std::cout << "==================================================\n\n";
        }

        // Apply the synchronized seed to ensure identical initial model weights
        torch::manual_seed(exchanged_seed);
        train_config_.random_seed = exchanged_seed;
    }

    void SetLogCallback(LogCallback callback) {
        net_log_callback = std::move(callback);
    }

    std::shared_ptr<IModel> GetFinalModel() {
		return local_engine_->GetFinalModel();
    }

    template<typename DatasetType>
    void Start(std::function<std::shared_ptr<IModel>()> model_builder, DatasetType user_data) {

        local_engine_ = std::make_unique<LocalTrainer>(model_builder, train_config_);

        // Instantiate the network balancer only on the Master node
        std::shared_ptr<NetworkBalancer> net_balancer = nullptr;
        if (net_config_.role == NetworkConfig::Role::Master) {
            net_balancer = std::make_shared<NetworkBalancer>(net_config_.world_size);
        }
        // The callback injected into LocalTrainer. Triggered after local gradients are computed.
        LocalTrainer::SyncCallback sync_callback = [this, net_balancer](const std::vector<float>& my_grads, double my_time, int my_samples) -> std::pair<std::vector<float>, EpochAssignment> {
            if (this->net_config_.role == NetworkConfig::Role::Worker) {
                // ---------- WORKER LOGIC ----------
                WorkerReport report{ this->net_config_.worker_id, static_cast<float>(my_time), my_samples };
                size_t msg_size = sizeof(WorkerReport) + my_grads.size() * sizeof(float);
                zmq::message_t req_msg(msg_size);

                // Serialize report and gradients into a single byte stream
                std::memcpy(req_msg.data(), &report, sizeof(WorkerReport));
                std::memcpy(static_cast<char*>(req_msg.data()) + sizeof(WorkerReport), my_grads.data(), my_grads.size() * sizeof(float));
                this->m_req_rep.send(req_msg, zmq::send_flags::none);

                // Wait for acknowledgment and the next global assignment
                zmq::message_t ack;
                this->m_req_rep.recv(ack, zmq::recv_flags::none);

                zmq::message_t sub_msg;
                this->m_pub_sub.recv(sub_msg, zmq::recv_flags::none);

                // Deserialize the global gradients and new batch indices
                std::vector<EpochAssignment> assignments(this->net_config_.world_size);
                size_t assign_bytes = assignments.size() * sizeof(EpochAssignment);
                std::memcpy(assignments.data(), sub_msg.data(), assign_bytes);
                std::vector<float> global_grads(my_grads.size());
                std::memcpy(global_grads.data(), static_cast<char*>(sub_msg.data()) + assign_bytes, global_grads.size() * sizeof(float));

                return { global_grads, assignments[this->net_config_.worker_id] };
            }
            else {
                // ---------- MASTER LOGIC ----------
                std::vector<WorkerReport> all_reports(this->net_config_.world_size);
                all_reports[0] = { 0, static_cast<float>(my_time), my_samples };

                // 1. Start with Master's OWN gradients, scaled by its OWN processed sample count
                std::vector<float> global_grads = my_grads;
                long long total_samples = my_samples; // Total number of samples processed across the entire cluster

                for (size_t j = 0; j < global_grads.size(); ++j) {
                    global_grads[j] *= my_samples;
                }

                // 2. Collect gradients from all connected Workers
                for (int i = 0; i < this->net_config_.expected_workers; ++i) {
                    zmq::message_t rep_msg;
                    this->m_req_rep.recv(rep_msg, zmq::recv_flags::none);

                    if (rep_msg.empty()) {
                        std::cerr << "\n[FATAL ERROR] Worker disconnected or Network Timeout (15s)!\n";
                        throw std::runtime_error("Network Timeout Error");
                    }

                    WorkerReport* worker_rep = static_cast<WorkerReport*>(rep_msg.data());
                    all_reports[worker_rep->worker_id] = *worker_rep;

                    int w_samples = worker_rep->processed_samples;
                    total_samples += w_samples; // Add the Worker's sample count to the global total

                    float* worker_grads = reinterpret_cast<float*>(static_cast<char*>(rep_msg.data()) + sizeof(WorkerReport));

                    // ADD Worker's gradients, scaled by ITS processed sample count
                    for (size_t j = 0; j < global_grads.size(); ++j) {
                        global_grads[j] += (worker_grads[j] * w_samples);
                    }

                    // Acknowledge receipt
                    std::string ack_str = "ACK";
                    zmq::message_t ack(ack_str.size());
                    std::memcpy(ack.data(), ack_str.data(), ack_str.size());
                    this->m_req_rep.send(ack, zmq::send_flags::none);
                }

                // 3. WEIGHTED AVERAGE: Divide the accumulated sum by the GLOBAL sample count
                if (total_samples > 0) {
                    for (size_t j = 0; j < global_grads.size(); ++j) {
                        global_grads[j] /= total_samples;
                    }
                }

                // 4. DYNAMIC NETWORK BALANCING
                // Master calculates the new data chunk ratios based on the collected execution times
                net_balancer->UpdateRatios(all_reports);
                std::vector<EpochAssignment> assignments = net_balancer->GenerateAssignments(this->train_config_.batch_size);

                // Serialize and broadcast the new assignments and the averaged global gradients
                size_t assign_bytes = assignments.size() * sizeof(EpochAssignment);
                size_t grads_bytes = global_grads.size() * sizeof(float);
                zmq::message_t pub_msg(assign_bytes + grads_bytes);

                std::memcpy(pub_msg.data(), assignments.data(), assign_bytes);
                std::memcpy(static_cast<char*>(pub_msg.data()) + assign_bytes, global_grads.data(), grads_bytes);
                this->m_pub_sub.send(pub_msg, zmq::send_flags::none);

                return { global_grads, assignments[0] };
            }
        };

        if (net_config_.world_size > 1) local_engine_->SetSyncCallback(sync_callback);

        if (net_config_.role == NetworkConfig::Role::Master) {
            std::cout << "[Master] Sync Threads (pause 1 sec)...\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cin >> std::ws;
            std::string cmd = "START_TRAINING";
            zmq::message_t pub_msg(cmd.size());
            std::memcpy(pub_msg.data(), cmd.data(), cmd.size());

            std::cout << "[Master] >>> Send Command To Start <<< \n";
            m_pub_sub.send(pub_msg, zmq::send_flags::none);
        }
        else {
            std::cout << "[Worker] Sleep. Waiting for start...\n";
            zmq::message_t sub_msg;
            m_pub_sub.recv(sub_msg, zmq::recv_flags::none); // The thread is physically blocked here until Master broadcasts
            std::cout << "[Worker] >>> Start working! <<< \n";
        }

        if (net_log_callback) {
            local_engine_->SetLogCallback(net_log_callback);
        }

        // All machines in the cluster exit this initialization block and start training simultaneously
        local_engine_->Fit(std::move(user_data));
    }
};
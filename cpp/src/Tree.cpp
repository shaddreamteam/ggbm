#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>
#include <random>
#include <numeric>

#include "Tree.h"
#include "TaskQueue.h"


class OptData;
void Tree::Construct(const TrainDataset& dataset,
                     const std::vector<float_type>& gradients,
                     const std::vector<float_type>& hessians,
                     std::vector<float_type>* current_predictions) {
    std::vector<uint32_t> indices  = SampleRows(dataset.GetRowCount());
    std::vector<Leaf> leafs = {Leaf(0, 0, dataset.GetFeatureCount(), indices)};
    std::vector<Leaf> parents;

    uint8_t depth;
    for(depth = 0; depth < config_.GetDepth(); ++depth) {
        std::vector<SearchParameters> split_params(dataset.GetFeatureCount());
        auto find_split = [&dataset, &leafs, &split_params,  &gradients, 
                           &hessians, &depth, &parents, this] 
                               (int32_t feature_number) {
            std::vector<Leaf>* parent_ptr = nullptr;
            if(depth > 0) {
                parent_ptr = &parents;
            }
            FindSplit(dataset, gradients, hessians, feature_number, depth,
                      parent_ptr,  &leafs, &split_params);
        };
        TaskQueue<decltype(find_split), int32_t> 
            task_queue(config_.GetThreads(), &find_split);
        for (uint32_t feature_number = 0;
             feature_number < dataset.GetFeatureCount();
             ++feature_number) {
            task_queue.Add(feature_number);
        }
        task_queue.Run();

        uint32_t best_feature = 0;
        float_type best_gain = 0;
        for (uint32_t i = 0; i < split_params.size(); ++i) {
            if (split_params[i].gain < best_gain) {
                best_feature = i;
                best_gain = split_params[i].gain;
            }
        }

        auto best_params = split_params[best_feature];
        for(uint32_t i = 0; i < leafs.size(); ++i) {
            float_type& left = best_params.left_weigths[i];
            float_type& right = best_params.right_weights[i];
            std::tie(left, right) = 
                 leafs[i].CalculateSplitWeights(best_feature,
                                                best_params.bin);
        }

        split_params.clear();
        splits_.emplace_back(best_feature, best_params.bin);
        parents.swap(leafs);
        leafs.clear();
        MakeNewLeafs(dataset.GetFeatureVector(best_feature), parents, best_params, leafs);
    }
    
    depth_ = depth;
    weights_ = std::vector<float_type>(uint32_t(pow(2, depth_)), 0);
    initialized_ = true;

    for(const auto& leaf: leafs) {
        auto index = leaf.GetIndex(depth_);
        if (index >= weights_.size()) {
            throw std::runtime_error("test");
        }
        weights_[leaf.GetIndex(depth_)] = leaf.GetWeight() * 
            config_.GetLearningRate();
    }

    for (uint32_t i = 0; i < leafs.size(); ++i) {
        for (auto row_index: leafs[i].GetRowIndices()) {
            (*current_predictions)[row_index] += weights_[i];
        }
    }
}


std::vector<float_type> Tree::PredictFromDataset(const Dataset& dataset) const{
    std::vector<float_type> predictions;
    predictions.reserve(dataset.GetRowCount());

    uint32_t base = pow(2, depth_);
    for(uint32_t row_number = 0; row_number < dataset.GetRowCount(); ++row_number) {
        uint32_t list_index = 0;
        for (auto& split : splits_) {
            if(dataset.GetFeature(row_number, split.feature) <= split.bin) {
                list_index = list_index * 2 + 1;
            } else {
                list_index = list_index * 2 + 2;
            }
        }
        auto prediction = weights_[list_index - base + 1];
        predictions.push_back(prediction);
    }
    return predictions;
}

bool Tree::IsInitialized() const {
    return initialized_;
}

void Tree::Save(std::ofstream& stream) {
    stream << splits_.size() << '\n';
    for (const auto& split: splits_) {
        stream << split.feature << " ";
        stream << static_cast<int32_t>(split.bin) << " ";
    }

    stream << "\n" << depth_ << "\n";
    stream << weights_.size() << "\n";
    for (auto weight: weights_) {
        stream << weight << " ";
    }
    stream << "\n";
}

void Tree::Load(std::ifstream& stream) {
    uint32_t split_count;
    stream >> split_count;
    splits_.resize(split_count);
    for (uint32_t split_number = 0; split_number < split_count; ++split_number) {
        uint32_t bin;
        stream >> splits_[split_number].feature >> bin;
        splits_[split_number].bin = static_cast<bin_id>(bin);
    }

    stream >> depth_;
    uint32_t weight_count;
    stream >> weight_count;
    weights_.resize(weight_count);
    for (uint32_t weight_number = 0; weight_number < weight_count; ++weight_number) {
        stream >> weights_[weight_number];
    }

    initialized_ = true;
}

std::vector<uint32_t> Tree::SampleRows(uint32_t n_rows) const {
    std::vector<uint32_t> row_indices;
    if(config_.GetRowSampling() >= 1 - EPS) {
        row_indices = std::vector<uint32_t>(n_rows);
        std::iota(row_indices.begin(), row_indices.end(), 0);
    } else {
        float_type sampling_coef = std::max(config_.GetRowSampling(),
                float(config_.GetMinSubsample()) / n_rows);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        for(uint32_t i = 0; i < n_rows; ++i) {
            if(distribution(generator) < sampling_coef) {
                row_indices.push_back(i);
            }
        }
    }
    return row_indices;
}

void Tree::MakeNewLeafs(const std::vector<bin_id>& feature_vector,
                        const std::vector<Leaf>& leafs,
                        const SearchParameters& best_params,
                        std::vector<Leaf>& new_leafs) {
    uint32_t new_leafs_size = leafs.size() * 2;
    new_leafs.resize(new_leafs_size);
    std::vector<uint32_t> indices_count(leafs.size());
    for (uint32_t leaf_index = 0; leaf_index < leafs.size(); ++leaf_index) {
        indices_count[leaf_index] = leafs[leaf_index].Size();
    }

    std::partial_sum(indices_count.begin(), indices_count.end(), indices_count.begin());

    std::vector<std::vector<std::vector<uint32_t>>> thread_leaf_indices(config_.GetThreads(),
                                               std::vector<std::vector<uint32_t>>(new_leafs_size));

    auto make_leafs = [&feature_vector, &leafs, &thread_leaf_indices,
            &indices_count, &best_params, this]
            (ThreadParameters thread_params) {
        uint32_t current_leaf = std::lower_bound(indices_count.begin(),
                                                 indices_count.end(),
                                                 thread_params.index_interval_start + 1) -
                                indices_count.begin();
        uint32_t previous_count = current_leaf == 0 ? 0 : indices_count[current_leaf - 1];
        uint32_t current_index = thread_params.index_interval_start - previous_count;
        uint32_t records_processed = 0;
        uint32_t records_to_process = thread_params.index_interval_end -
                thread_params.index_interval_start;
        std::vector<std::vector<uint32_t>>& thread_leaf_rows =
                thread_leaf_indices[thread_params.thread_id];

        while (records_processed < records_to_process && current_leaf < leafs.size()) {
            uint32_t current_leaf_size = leafs[current_leaf].Size();
            auto row_indices = leafs[current_leaf].GetRowIndices();
            while (current_index < current_leaf_size && records_processed < records_to_process) {
                uint32_t row_index = row_indices[current_index];
                if (feature_vector[row_index] <= best_params.bin) {
                    // left
                    thread_leaf_rows[current_leaf * 2].push_back(row_index);
                } else {
                    thread_leaf_rows[current_leaf * 2 + 1].push_back(row_index);
                }
                ++records_processed;
                ++current_index;
            }
            current_index = 0;
            ++current_leaf;
        }
    };

    TaskQueue<decltype(make_leafs), ThreadParameters>
            leafs_queue(config_.GetThreads(), &make_leafs);
    uint32_t start = 0;
    for (uint32_t part_number = 1; part_number <= config_.GetThreads(); ++part_number) {
        ThreadParameters thread_params(start,
                                       feature_vector.size() * part_number / config_.GetThreads(),
                                       part_number - 1);
        leafs_queue.Add(thread_params);
        // end is not included
        start = thread_params.index_interval_end;
    }
    leafs_queue.Run();

    auto fill_leafs = [&new_leafs, &leafs, &thread_leaf_indices,
            &indices_count, &best_params, this]
            (uint32_t leaf_number) {
        uint32_t size = 0;
        for (uint32_t i = 0; i < thread_leaf_indices.size(); ++i) {
            size += thread_leaf_indices[i][leaf_number].size();
        }
        new_leafs[leaf_number].row_indices_.reserve(size);
        for (uint32_t i = 0; i < thread_leaf_indices.size(); ++i) {
            for (uint32_t j = 0; j < thread_leaf_indices[i][leaf_number].size(); ++j) {
                new_leafs[leaf_number].row_indices_.push_back(
                        thread_leaf_indices[i][leaf_number][j]);
            }
        }

        uint32_t is_right = leaf_number % 2;
        new_leafs[leaf_number].leaf_index_ = leafs[leaf_number / 2].leaf_index_ * 2 + 1 + is_right;
        if (new_leafs[leaf_number].IsEmpty()) {
            new_leafs[leaf_number].weight_ = leafs[leaf_number / 2].weight_;
        } else {
            if (is_right) {
                new_leafs[leaf_number].weight_ = best_params.right_weights[leaf_number / 2];
            } else {
                new_leafs[leaf_number].weight_ = best_params.left_weigths[leaf_number / 2];
            }
            new_leafs[leaf_number].histograms_.resize(leafs[leaf_number / 2].histograms_.size());
        }
    };

    TaskQueue<decltype(fill_leafs), uint32_t>
            fill_queue(std::min(config_.GetThreads(), new_leafs_size), &fill_leafs);
    for(uint32_t leaf_number = 0; leaf_number < new_leafs_size; ++leaf_number) {
        fill_queue.Add(leaf_number);
    }
    fill_queue.Run();
}

void Tree::FindSplit(const TrainDataset& dataset,
                     const std::vector<float_type>& gradients,
                     const std::vector<float_type>& hessians,
                     uint32_t feature_number,
                     uint32_t depth,
                     const std::vector<Leaf>* parent_leafs,
                     std::vector<Leaf>* leafs,
                     std::vector<SearchParameters>* split_params) const {
    const std::vector<bin_id>& feature_vector =
        dataset.GetFeatureVector(feature_number);
    uint32_t bin_count = dataset.GetBinCount(feature_number);
    SearchParameters search_parameters;
    search_parameters.left_weigths = std::vector<float_type>(leafs->size(), 0);
    search_parameters.right_weights= std::vector<float_type>(leafs->size(), 0);
    uint32_t base = uint32_t(pow(2, depth - 1));

    if(!parent_leafs) {
        for(Leaf& leaf : *leafs) {
            leaf.CalculateHistogram(
                feature_number, config_.GetLambdaL2(), bin_count,
                feature_vector, gradients, hessians);
        }
    } else {
        for(uint32_t left_index = 0; left_index < (*leafs).size();
                left_index += 2) {
            Leaf* smaller = &(*leafs)[left_index];
            Leaf* bigger = &(*leafs)[left_index + 1];
            if(smaller->Size() > bigger->Size()) {
                smaller = &(*leafs)[left_index + 1];
                bigger = &(*leafs)[left_index];
            }
            
            uint32_t parent_index = bigger->ParentVectorIndex(base);
            const Leaf& parent = (*parent_leafs)[parent_index];
            if(smaller->Size() != 0) {
                smaller->CalculateHistogram(
                    feature_number, config_.GetLambdaL2(), bin_count,
                    feature_vector, gradients, hessians);
                bigger->DiffHistogram(feature_number, parent, *smaller);
            } else {
                if(bigger->Size() > 0) {
                    bigger->CopyHistogram(feature_number, parent);
                }
            }
        }
    }

    for(bin_id bin_number = 0; bin_number < bin_count; ++bin_number) {
        float_type gain = 0;
        for(uint32_t leaf_number = 0; leaf_number < leafs->size();
                ++leaf_number) {
            gain += (*leafs)[leaf_number].CalculateSplitGain(feature_number,
                                                             bin_number);
        }

        if(gain <= search_parameters.gain) {
            search_parameters.gain = gain;
            search_parameters.bin = bin_number;
        }
    }

    (*split_params)[feature_number] = search_parameters;
};

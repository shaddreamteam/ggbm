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
                     const std::vector<float_type>& hessians) {
    std::vector<uint32_t> indexes  = SampleRows(dataset.GetRowCount());
    std::vector<Leaf> leafs = {Leaf(0, 0, dataset.GetFeatureCount(), indexes)};
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
        for (int32_t feature_number = 0;
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
        split_params.clear();
        splits_.emplace_back(best_feature, best_params.bin);
        parents.swap(leafs);
        leafs = MakeNewLeafs(dataset.GetFeatureVector(best_feature),
                             parents, best_params);
    }
    
    depth_ = depth;
    weights_ = std::vector<float_type>(uint32_t(pow(2, depth_)), 0);
    initialized_ = true;

    for(const auto& leaf: leafs) {
        weights_[leaf.GetIndex(depth_)] = leaf.GetWeight() * 
            config_.GetLearningRate();
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
        /*if (list_index - uint32_t(pow(2, depth_)) + 1 >= weights_.size() ||
                list_index - uint32_t(pow(2, depth_)) + 1 < 0) {
            std::cout << list_index - uint32_t(pow(2, depth_)) + 1 << std::endl;
        }*/
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
    for (int32_t split_number = 0; split_number < split_count; ++split_number) {
        int32_t bin;
        stream >> splits_[split_number].feature >> bin;
        splits_[split_number].bin = static_cast<bin_id>(bin);
    }

    stream >> depth_;
    uint32_t weight_count;
    stream >> weight_count;
    weights_.resize(weight_count);
    for (int32_t weight_number = 0; weight_number < weight_count; ++weight_number) {
        stream >> weights_[weight_number];
    }

    initialized_ = true;
}

std::vector<uint32_t> Tree::SampleRows(uint32_t n_rows) const {
    std::vector<uint32_t> row_indexes;
    if(config_.GetRowSampling() >= 1 - EPS) {
        row_indexes = std::vector<uint32_t>(n_rows);
        std::iota(row_indexes.begin(), row_indexes.end(), 0);
    } else {
        float_type sampling_coef = std::max(config_.GetRowSampling(),
                float(config_.GetMinSubsample()) / n_rows);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        for(uint32_t i = 0; i < n_rows; ++i) {
            if(distribution(generator) < sampling_coef) {
                row_indexes.push_back(i);
            }
        }
    }
    return row_indexes;
}

std::vector<Leaf> Tree::MakeNewLeafs(std::vector<bin_id> feature_vector,
                                     const std::vector<Leaf>& leafs,
                                     const SearchParameters& best_params) {
    Leaf left, right;
    std::vector<Leaf> new_leafs;
    for(uint32_t leaf_number = 0; leaf_number < leafs.size(); ++leaf_number) {
        const Leaf& leaf = leafs[leaf_number];
        float_type child_weight = best_params.left_weigths[leaf_number];
        new_leafs.push_back(leaf.MakeChild(
            true, feature_vector, best_params.bin, child_weight));

        child_weight = best_params.right_weights[leaf_number];
        new_leafs.push_back(leaf.MakeChild(
            false, feature_vector, best_params.bin, child_weight));
    }
    return new_leafs;
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

        std::vector<float_type> left_weigths(leafs->size());
        std::vector<float_type> right_weigts(leafs->size());
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
            for(uint32_t i = 0; i < leafs->size(); ++i) {
                float_type& left = search_parameters.left_weigths[i];
                float_type& right = search_parameters.right_weights[i];
                std::tie(left, right) = 
                     (*leafs)[i].CalculateSplitWeights(feature_number,
                                                       bin_number);
            }
        }
    }

    (*split_params)[feature_number] = search_parameters;
};

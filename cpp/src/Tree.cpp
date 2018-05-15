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
void Tree::Construct(Dataset* dataset) {
    auto data = dataset->GetData();
    std::vector<uint32_t> indices  = SampleRows(dataset->GetRowCount());
    std::vector<Leaf> leafs;
    leafs.emplace_back(0,
                       0,
                       dataset->GetFeatureCount() * config_.GetThreads(),
                       indices,
                       dataset->GetBinCounts(),
                       config_.GetLambdaL2());

    // TODO Parallel
    for (uint32_t i = 0; i < indices.size(); ++i) {
        data[indices[i]].leaf_index = 0;
    }
    std::vector<Leaf> parents;

    uint8_t depth;
    for(depth = 0; depth < config_.GetDepth(); ++depth) {
        std::vector<SearchParameters> split_params(dataset->GetFeatureCount());
        auto calc_hists = [&dataset, &indices, &leafs, this]
                               (ThreadParameters thread_params) {
            auto data = dataset->GetData();

            for (uint32_t current_index = thread_params.index_interval_start;
                 current_index < thread_params.index_interval_end;
                 ++current_index) {
                auto row = data[indices[current_index]];
                // TODO without row
                auto histograms = leafs[row.leaf_index].GetHistograms();
                auto feature_count = dataset->GetFeatureCount();
                uint32_t hist_offset = thread_params.thread_id * feature_count;
                for (uint32_t feature_number = 0;
                     feature_number < feature_count;
                     ++feature_number) {
                    //if (hist_offset + feature_number >= 4 * feature_count) {
                    //    throw std::runtime_error("test");
                    //}
                    histograms[hist_offset + feature_number].AddGradientAndHessian(
                            row.bin_ids[feature_number],
                            row.gradient,
                            row.hessian);
                }
            }
        };

        TaskQueue<decltype(calc_hists), ThreadParameters>
            hist_queue(config_.GetThreads(), &calc_hists);
        uint32_t start = 0;
        for (uint32_t part_number = 1; part_number <= config_.GetThreads(); ++part_number) {
            ThreadParameters thread_params(part_number - 1,
                                           start,
                                           (indices.size() * part_number) / config_.GetThreads());
            hist_queue.Add(thread_params);
            // end is not included
            start = thread_params.index_interval_end;
        }
        hist_queue.Run();

        auto find_split = [&dataset, &indices, &leafs, &split_params, &depth, this]
                (uint32_t feature_number) {
            uint32_t hist_offset = dataset->GetFeatureCount();
            for (auto& leaf: leafs) {
                if (leaf.IsEmpty()) {
                    continue;
                }
                auto histograms = leaf.GetHistograms();
                auto first = histograms[feature_number];
                for (uint32_t thread_id = 1; thread_id < config_.GetThreads(); ++thread_id) {
                    first += histograms[feature_number + hist_offset * thread_id];
                }

                for(uint32_t i = 1; i < first.bin_count; ++i) {
                    first.gradients_hessians[i].gradient +=
                            first.gradients_hessians[i - 1].gradient;
                    first.gradients_hessians[i].hessian +=
                            first.gradients_hessians[i - 1].hessian;
                }
            }

            SearchParameters search_parameters;
            auto bin_count = dataset->GetBinCounts()[feature_number];
            for(bin_id bin_number = 0; bin_number < bin_count; ++bin_number) {
                float_type gain = 0;
                for(uint32_t leaf_number = 0; leaf_number < leafs.size(); ++leaf_number) {
                    gain += leafs[leaf_number].CalculateSplitGain(feature_number, bin_number);
                }

                if(gain <= search_parameters.gain) {
                    search_parameters.gain = gain;
                    search_parameters.bin = bin_number;
                }
            }

            split_params[feature_number] = search_parameters;
        };

        TaskQueue<decltype(find_split), uint32_t>
                split_queue(config_.GetThreads(), &find_split);
        for (uint32_t feature = 0; feature < dataset->GetFeatureCount(); ++feature) {
            split_queue.Add(feature);
        }
        split_queue.Run();

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
            best_params.left_weigths.resize(leafs.size());
            best_params.right_weights.resize(leafs.size());
            float_type& left = best_params.left_weigths[i];
            float_type& right = best_params.right_weights[i];
            std::tie(left, right) =
                    leafs[i].CalculateSplitWeights(best_feature, best_params.bin);
        }

        splits_.emplace_back(best_feature, best_params.bin);
        parents.swap(leafs);
        leafs.clear();
        MakeNewLeafs(dataset, parents, best_feature, best_params, &leafs, depth);
    }
    
    depth_ = depth;
    weights_ = std::vector<float_type>(uint32_t(pow(2, depth_)), 0);
    initialized_ = true;

    for(const auto& leaf: leafs) {
        weights_[leaf.GetIndex(depth_)] = leaf.GetWeight() * 
            config_.GetLearningRate();
    }
}


void Tree::UpdatePredictions(Dataset* dataset) const {
    auto predict = [&dataset, this]
            (ThreadParameters thread_params) {
        auto data = dataset->GetData();

        uint32_t base = pow(2, depth_);
        for(uint32_t row_number = thread_params.index_interval_start;
            row_number < thread_params.index_interval_end;
            ++row_number) {
            uint32_t list_index = 0;
            for (auto& split : splits_) {
                if(data[row_number].bin_ids[split.feature] <= split.bin) {
                    list_index = list_index * 2 + 1;
                } else {
                    list_index = list_index * 2 + 2;
                }
            }
            data[row_number].prediction += weights_[list_index - base + 1];
        }
    };


    TaskQueue<decltype(predict), ThreadParameters> predict_queue(config_.GetThreads(), &predict);
    uint32_t parts = dataset->GetRowCount() / config_.GetThreads();
    uint32_t start = 0;
    for (uint32_t part_number = 1; part_number <= parts; ++part_number) {
        ThreadParameters thread_params(part_number - 1,
                                       start,
                                       (dataset->GetRowCount() * part_number) / parts);
        predict_queue.Add(thread_params);
        // end is not included
        start = thread_params.index_interval_end;
    }
    predict_queue.Run();
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
    if(config_.GetRowSampling() >= 1 - kEps) {
        row_indexes = std::vector<uint32_t>(n_rows);
        std::iota(row_indexes.begin(), row_indexes.end(), 0);
    } else {
        float_type sampling_coef = std::max(config_.GetRowSampling(),
                float(config_.GetMinSubsample()) / n_rows);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for(uint32_t i = 0; i < n_rows; ++i) {
            if(distribution(generator) < sampling_coef) {
                row_indexes.push_back(i);
            }
        }
    }
    return row_indexes;
}

void Tree::MakeNewLeafs(Dataset* dataset,
                        const std::vector<Leaf>& leafs,
                        uint32_t feature_number,
                        const SearchParameters& best_params,
                        std::vector<Leaf>* children,
                        uint32_t depth) {
    //std::vector<Leaf> new_leafs;
    for(uint32_t leaf_number = 0; leaf_number < leafs.size(); ++leaf_number) {
        const Leaf& leaf = leafs[leaf_number];
        float_type child_weight = best_params.left_weigths[leaf_number];
        children->emplace_back(leaf.MakeChild(true,
                                           dataset,
                                           feature_number,
                                           best_params.bin,
                                           child_weight,
                                              depth));

        child_weight = best_params.right_weights[leaf_number];
        children->emplace_back(leaf.MakeChild(false,
                                              dataset,
                                              feature_number,
                                              best_params.bin,
                                              child_weight,
                                              depth));
    }
    //return std::move(new_leafs);
}

void Tree::FindSplit(Dataset* dataset,
                     const std::vector<uint32_t>& indices,
                     ThreadParameters thread_params,
                     uint32_t depth,
                     const std::vector<Leaf>* parent_leafs,
                     std::vector<Leaf>* leafs,
                     std::vector<SearchParameters>* split_params) {


    /*uint32_t bin_count = dataset.GetBinCount(feature_number);
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

    (*split_params)[feature_number] = search_parameters; */
};

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
#include <random>
#include <numeric>

#include "Tree.h"
#include "TaskQueue.h"

struct SearchParameters {
    SearchParameters() : gain(0), bin(0) {}

    float_type gain;
    bin_id bin;
    std::vector<float_type> left_weigths;
    std::vector<float_type> right_weights;
};

class OptData;
void Tree::Construct(const TrainDataset& dataset,
                     const std::vector<float_type>& gradients,
                     const std::vector<float_type>& hessians,
                     float_type lambda_l2_reg,
                     float_type row_sampling,
                     uint32_t min_subsample) {
    std::vector<uint32_t> indexes;
    // Here we should check that row_sampling in (0, 1]
    // and min_subsample <= dataset-GetRowCount
    if(row_sampling >= 1 - EPS) {
        indexes = std::vector<uint32_t>(dataset.GetRowCount());
        std::iota(indexes.begin(), indexes.end(), 0);
    } else {
        float_type sampling_coef = std::max(row_sampling,
                float(min_subsample) / dataset.GetRowCount());
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
            if(distribution(generator) < sampling_coef) {
                indexes.push_back(i);
            }
        }
    }

    std::vector<Leaf> leafs = {Leaf(0, 0, indexes)};

    for(depth_ = 0; depth_ < max_depth_; ++depth_) {
        std::vector<SearchParameters> split_params(dataset.GetFeatureCount());

        auto find_split = [&dataset, &leafs, &split_params,
                lambda_l2_reg, gradients, hessians, this](int32_t feature_number) {
            const std::vector<bin_id>& feature_vector = 
                dataset.GetFeatureVector(feature_number);
            uint32_t bin_count = dataset.GetBinCount(feature_number);
            SearchParameters search_parameters;
            for(bin_id bin_number = 0; bin_number < bin_count; ++bin_number) {
                float_type gain = 0;
                std::vector<float_type> left_weigths(leafs.size()), right_weigts(leafs.size());
                std::vector<Histogram> histograms;
                for(const Leaf& leaf : leafs) {
                    histograms.push_back(leaf.GetHistogram(feature_number,
                                                           lambda_l2_reg,
                                                           bin_count,
                                                           feature_vector,
                                                           gradients,
                                                           hessians));
                }
                for(uint32_t hist_number = 0; hist_number < histograms.size(); ++hist_number) {
                    gain += histograms[hist_number].CalculateSplitGain(bin_number);
                    std::tie(left_weigths[hist_number],
                             right_weigts[hist_number]) =
                            histograms[hist_number].CalculateSplitWeights(bin_number);
                }

                if(gain <= search_parameters.gain) {
                    search_parameters.gain = gain;
                    search_parameters.bin = bin_number;
                    search_parameters.left_weigths = left_weigths;
                    search_parameters.right_weights = right_weigts;
                }
            }

            split_params[feature_number] = search_parameters;
        };

        TaskQueue<decltype(find_split), int32_t> task_queue(thread_count_, &find_split);
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

        splits_.push_back(std::make_tuple(best_feature, best_params.bin));
        Leaf left, right;
        std::vector<Leaf> new_leafs;
        const std::vector<bin_id>& feature_vector = 
            dataset.GetFeatureVector(best_feature);
        for(uint32_t leaf_number = 0; leaf_number < leafs.size(); ++leaf_number) {
            const Leaf& leaf = leafs[leaf_number];
            float_type child_weight = best_params.left_weigths[leaf_number];
            new_leafs.push_back(leaf.MakeChild(
                true, feature_vector, best_params.bin, child_weight));

            child_weight = best_params.right_weights[leaf_number];
            new_leafs.push_back(leaf.MakeChild(
                false, feature_vector, best_params.bin, child_weight));
        }
        leafs = new_leafs;
    }
    
    weights_ = std::vector<float_type>(uint32_t(pow(2, depth_)), 0);
    initialized_ = true;

    for(const auto& leaf: leafs) {
        weights_[leaf.GetIndex(depth_)] = leaf.GetWeight();
    }
}


std::vector<float_type> Tree::PredictFromDataset(const Dataset& dataset) const{
    std::vector<float_type> predictions;
    predictions.reserve(dataset.GetRowCount());

    for(uint32_t object_index = 0; object_index < dataset.GetRowCount(); ++object_index) {
        uint32_t list_index = 0;
        for (auto& split : splits_) {
            if(dataset.GetFeature(object_index, std::get<0>(split)) <= std::get<1>(split)) {
                list_index = list_index * 2 + 1;
            } else {
                list_index = list_index * 2 + 2;
            }
        }
        auto prediction = weights_[list_index - uint32_t(pow(2, depth_)) + 1];
        predictions.push_back(prediction);
    }
    return predictions;
}


std::vector<float_type> Tree::PredictFromFile(const std::string& filename,
                                              const FeatureTransformer& ft,
                                              bool fileHasTarget, 
                                              char sep) const {
    TestDataset dataset(filename, ft, fileHasTarget);
    return PredictFromDataset(dataset);
}

bool Tree::IsInitialized() const {
    return initialized_;
}

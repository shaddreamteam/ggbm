#include <algorithm>
#include <cmath>
//#include <function>
//#include <functional>
#include <tuple>
#include <vector>
#include <random>
#include <numeric>
#include "Tree.h"

class OptData;
void Tree::Construct(std::shared_ptr<const TrainDataset> dataset,
                     std::shared_ptr<const OptData> optData,
                     float_type lambda_l2_reg,
                     float_type row_sampling,
                     uint32_t min_subsample) {
    std::vector<uint32_t> indexes;
    // Here we should check that row_sampling in (0, 1]
    // and min_subsample <= dataset-GetNRows
    if(row_sampling >= 1 - EPS) {
        indexes = std::vector<uint32_t>(dataset->GetNRows());
        std::iota(indexes.begin(), indexes.end(), 0);
    } else {
        float_type sampling_coef = std::max(row_sampling,
                float(min_subsample) / dataset->GetNRows());
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        for(uint32_t i = 0; i < dataset->GetNRows(); ++i) {
            if(distribution(generator) < sampling_coef) {
                indexes.push_back(i);
            }
        }
    }

    std::vector<std::tuple<uint32_t, bin_id>> splits;
    std::vector<Leaf> leafs = {Leaf(0, 0, indexes, dataset, optData)};;
    std::vector<Leaf> best_leafs = leafs;
    uint32_t best_depth = 0;
    float_type best_gain = 0;

    for(uint32_t depth = 0; depth < max_depth_; ++depth) {
        float_type prev_gain = best_gain;
        uint32_t best_feature;
        bin_id best_bin;
        uint32_t size = leafs.size();
        std::vector<float_type> best_left_weigths(size), best_right_weights(size);
        for(uint32_t feature_number = 0; feature_number < dataset->GetNFeatures(); ++feature_number) {
            for(bin_id bin_number = 0; bin_number < dataset->GetBinCount(feature_number); ++bin_number) {
                float_type gain = 0;
                std::vector<float_type> left_weigths(size), right_weigts(size);
                std::vector<Histogram> histograms;
                for(const Leaf& leaf : leafs) {
                    histograms.push_back(leaf.GetHistogram(feature_number, lambda_l2_reg));
                }
                for(uint32_t hist_number = 0; hist_number < histograms.size(); ++hist_number) {
                    gain += histograms[hist_number].CalculateSplitGain(bin_number); 
                    std::tie(left_weigths[hist_number], 
                            right_weigts[hist_number]) =
                        histograms[hist_number].CalculateSplitWeights(bin_number);
                }

                if(gain < best_gain) {
                    best_gain = gain;
                    best_feature = feature_number;
                    best_bin = bin_number;
                    best_left_weigths = left_weigths;
                    best_right_weights = right_weigts;
                }
            }
        }

        splits.push_back(std::make_tuple(best_feature, best_bin));
        Leaf left, right;
        std::vector<Leaf> new_leafs;
        for(uint32_t leaf_number = 0; leaf_number < leafs.size(); ++leaf_number) {
            std::tie(left, right) = leafs[leaf_number].MakeChilds(best_feature, best_bin,
                    best_left_weigths[leaf_number], best_right_weights[leaf_number]);
            if(!left.IsEmpty()) {
                new_leafs.push_back(left);
            }
            if(!right.IsEmpty()) {
                new_leafs.push_back(right);
            }
        }
        leafs = new_leafs;

        if(best_gain < prev_gain - EPS) {
            best_leafs = new_leafs;
            best_depth = depth + 1;
        }
    }
    
    if(best_depth > 0) {
        depth_ = best_depth;
        weights_ = std::vector<float_type>(uint32_t(pow(2, depth_)), 0);
        initialized_ = true;

        for(const auto& leaf : best_leafs) {
            weights_[leaf.GetIndex(depth_)] = leaf.GetWeight();
        }
        for(uint32_t i = 0; i < best_depth; ++ i) {
            splits_.push_back(splits[i]);
        }
    }
}


std::vector<float_type> Tree::PredictFromDataset(const Dataset& dataset) const{
    std::vector<float_type> predictions;
    predictions.reserve(dataset.GetNRows());

    for(uint32_t object_index = 0; object_index < dataset.GetNRows(); ++object_index) {
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

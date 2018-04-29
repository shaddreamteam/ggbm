#include <algorithm>
#include <cmath>
//#include <function>
//#include <functional>
#include <tuple>
#include <vector>
#include <numeric>
#include "Tree.h"

void Tree::Construct(std::shared_ptr<const Dataset> dataset, std::shared_ptr<const OptData> optData, float_type lambda_l2_reg) {
    std::vector<uint32_t> indexes(dataset->GetNRows());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::vector<Leaf> leafs = {Leaf(0, 0, indexes, dataset, optData)};;
    float_type best_gain = 0;
    uint32_t depth;
    for(depth = 0; depth < max_depth_; ++depth) {
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

        if(best_gain < prev_gain - EPS) {
            splits_.push_back(std::make_tuple(best_feature, best_bin));
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
            ++depth_;
        } else {
            break;
        }
    }
    
    if(depth > 0) {
        weights_ = std::vector<float_type>(uint32_t(pow(2, depth)), 0);
        initialized_ = true;
    }

    for(const auto& leaf : leafs) {
        weights_[leaf.GetIndex(depth)] = leaf.GetWeight();
    }
}


std::vector<float_type> Tree::PredictFromBins(const std::vector<std::vector<bin_id>>& data_x_binned) const{
    std::vector<float_type> predictions;
    predictions.reserve(weights_.size());

    for(uint32_t object_index = 0; object_index < data_x_binned[0].size(); ++object_index) {
        uint32_t list_index = 0;
        for (auto& split : splits_) {
            if(data_x_binned[std::get<0>(split)][object_index] <= std::get<1>(split)) {
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
                                              std::shared_ptr<const Dataset> dataset,
                                              char sep) const {
    std::vector<std::vector<float_type>> data_x;
    dataset->GetSampleAndTargetFromFile(filename, sep, &data_x); 
    return PredictFromBins(dataset->Transform(data_x));
}

#include "Leaf.h"
#include <cmath>


float_type Histogram::CalculateGain(float_type gradient, float_type hessian, 
                                     uint32_t row_count) const {
    float_type denominator = (hessian + row_count * lambda_l2_reg_);
    float_type gain = 0;
    if(denominator != 0) {
        gain = -pow(gradient, 2.0) / (2.0 * denominator);
    }
    return gain;
}

float_type Histogram::CalculateWeight(float_type gradient, float_type hessian, 
                                      uint32_t row_count) const {
    float_type denominator = (hessian + row_count * lambda_l2_reg_);
    float_type weight = 0;
    if(denominator != 0) {
        weight = -pow(gradient, 2.0) / (2.0 * denominator);
    }
    return weight;
}

std::tuple<float_type, float_type> Histogram::CalculateGains(bin_id bin_number) const {
    float_type gain_left = CalculateGain(gradient_cs_[bin_number], hessian_cs_[bin_number],
                                         row_count_cs_[bin_number]);
    float_type gain_right = CalculateGain(gradient_cs_.back() -  gradient_cs_[bin_number], 
                                         hessian_cs_.back() -  hessian_cs_[bin_number],
                                         row_count_cs_.back() -  row_count_cs_[bin_number]);
    return std::make_tuple(gain_left, gain_right);
}

std::tuple<float_type, float_type> Histogram::CalculateWeights(bin_id bin_number) const {
    float_type weght_left = CalculateWeight(gradient_cs_[bin_number], hessian_cs_[bin_number],
                                            row_count_cs_[bin_number]);
    float_type weight_right = CalculateWeight(gradient_cs_.back() -  gradient_cs_[bin_number], 
                                              hessian_cs_.back() -  hessian_cs_[bin_number],
                                              row_count_cs_.back() -  row_count_cs_[bin_number]);
    return std::make_tuple(weght_left, weight_right);
}

Histogram Leaf::GetWeightsGains(uint32_t feature_number) {
    std::vector<float_type> gradients(dataset.GetBinCount(feature_number), 0);
    std::vector<float_type> hessians(dataset.GetBinCount(feature_number), 0);
    std::vector<uint32_t> row_counts(dataset.GetBinCount(feature_number), 0);
    for(uint32_t index : indexes) {
        bin_id bin_number = dataset.GetFeature(index, feature_number);
        gradients[bin_number] += optData.GetGradient(index);
        hessians[bin_number] += optData.GetHessian(index);
        row_counts[bin_number] += 1;
    }
    for(uint32_t i = 1; i < dataset.GetBinCount(feature_number); ++i) {
        gradients[i] += gradients[i - 1];
        hessians[i] += hessians[i - 1];
        row_counts[i] += row_counts[i - 1];
    }
    return Histogram(gradients, hessians, row_counts, lambda_l2_reg_);
}

#include "Leaf.h"
#include <cmath>


float_type Histogram::CalculateGain(float_type gradient,
                                    float_type hessian) const {
    float_type denominator = hessian + lambda_l2_reg_;
    float_type gain = 0;
    if(denominator != 0) {
        gain = -pow(gradient, 2.0) / (2.0 * denominator);
    }
    return gain;
}

float_type Histogram::CalculateWeight(float_type gradient,
                                      float_type hessian) const {
    float_type denominator = hessian + lambda_l2_reg_;
    float_type weight = 0;
    if(denominator != 0) {
        weight = -gradient / denominator;
    }
    return weight;
}

float_type Histogram::CalculateSplitGain(bin_id bin_number) const {
    if(empty_leaf_) {
        return 0;
    }
    float_type gain_left = CalculateGain(gradient_cs_[bin_number],
                                         hessian_cs_[bin_number]);

    float_type right_gradient_sum = gradient_cs_.back() - 
        gradient_cs_[bin_number];
    float_type right_hessian_sum = hessian_cs_.back() -
        hessian_cs_[bin_number];
    float_type gain_right = CalculateGain(right_gradient_sum,
                                          right_hessian_sum);

    return gain_left + gain_right;
}

std::tuple<float_type, float_type> Histogram::CalculateSplitWeights(
        bin_id bin_number) const {
    if(empty_leaf_) {
        return std::make_tuple(0, 0);
    }
    float_type weght_left = CalculateWeight(gradient_cs_[bin_number],
                                            hessian_cs_[bin_number]);

    float_type right_gradient_sum = gradient_cs_.back() - 
        gradient_cs_[bin_number];
    float_type right_hessian_sum = hessian_cs_.back() -
        hessian_cs_[bin_number];
    float_type weight_right = CalculateWeight(right_gradient_sum, 
                                              right_hessian_sum);

    return std::make_tuple(weght_left, weight_right);
}

Histogram Leaf::GetHistogram(uint32_t feature_number,
                             float_type lambda_l2_reg,
                             uint32_t bin_count,
                             const std::vector<bin_id>& feature_vector,
                             const std::vector<float_type>& gradients,
                             const std::vector<float_type>& hessians) const {
    if(row_indexes_.empty()) {
        return Histogram(std::vector<float_type>(0),
                         std::vector<float_type>(0),
                         lambda_l2_reg, true);
    }
    std::vector<float_type> gradient_sum(bin_count, 0);
    std::vector<float_type> hessian_sum(bin_count, 0);
    for(uint32_t index : row_indexes_) {
        bin_id bin_number = feature_vector[index];
        gradient_sum[bin_number] += gradients[index];
        hessian_sum[bin_number] += hessians[index];
    }
    for(uint32_t i = 1; i < bin_count; ++i) {
        gradient_sum[i] += gradient_sum[i - 1];
        hessian_sum[i] += hessian_sum[i - 1];
    }
    return Histogram(gradient_sum, hessian_sum, lambda_l2_reg, false);
}

Leaf Leaf::MakeChild(bool is_left, const std::vector<bin_id>& feature_vector,
                     bin_id bin_number, float_type weight) const {
    std::vector<uint32_t> child_rows(0);
    for(uint32_t index : row_indexes_) {
        bool belongs_left = feature_vector[index] <= bin_number;
        if(belongs_left && is_left || !belongs_left && !is_left) {
            child_rows.push_back(index);
        }
    }

    if(child_rows.empty()) {
        weight = weight_;
    }

    uint32_t child_index;
    if(is_left) {
        child_index = leaf_index_ * 2 + 1;
    } else {
        child_index = leaf_index_ * 2 + 2;
    }

    Leaf child(child_index, weight, child_rows);
    return child;
}

uint32_t Leaf::GetIndex(uint32_t depth) const {
    return leaf_index_ - uint32_t(pow(2, depth)) + 1;
}

bool Leaf::IsEmpty() const {
    return row_indexes_.empty();
}

float_type Leaf::GetWeight() const {
    return weight_;
}

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
        weight = -gradient / denominator;
    }
    return weight;
}

float_type Histogram::CalculateSplitGain(bin_id bin_number) const {
    float_type gain_left = CalculateGain(gradient_cs_[bin_number], hessian_cs_[bin_number],
                                         row_count_cs_[bin_number]);
    float_type gain_right = CalculateGain(gradient_cs_.back() -  gradient_cs_[bin_number], 
                                         hessian_cs_.back() -  hessian_cs_[bin_number],
                                         row_count_cs_.back() -  row_count_cs_[bin_number]);
    return gain_left + gain_right;
}

std::tuple<float_type, float_type> Histogram::CalculateSplitWeights(bin_id bin_number) const {
    float_type weght_left = CalculateWeight(gradient_cs_[bin_number], hessian_cs_[bin_number],
                                            row_count_cs_[bin_number]);
    float_type weight_right = CalculateWeight(gradient_cs_.back() -  gradient_cs_[bin_number], 
                                              hessian_cs_.back() -  hessian_cs_[bin_number],
                                              row_count_cs_.back() -  row_count_cs_[bin_number]);
    return std::make_tuple(weght_left, weight_right);
}

Histogram Leaf::GetHistogram(uint32_t feature_number, float_type lambda_l2_reg) const {
    std::vector<float_type> gradients(dataset_->GetBinCount(feature_number), 0);
    std::vector<float_type> hessians(dataset_->GetBinCount(feature_number), 0);
    std::vector<uint32_t> row_counts(dataset_->GetBinCount(feature_number), 0);
    for(uint32_t index : row_indexes_) {
        bin_id bin_number = dataset_->GetFeature(index, feature_number);
        gradients[bin_number] += optData_->GetGradient(index);
        hessians[bin_number] += optData_->GetHessian(index);
        row_counts[bin_number] += 1;
    }
    for(uint32_t i = 1; i < dataset_->GetBinCount(feature_number); ++i) {
        gradients[i] += gradients[i - 1];
        hessians[i] += hessians[i - 1];
        row_counts[i] += row_counts[i - 1];
    }
    return Histogram(gradients, hessians, row_counts, lambda_l2_reg);
}

std::tuple<Leaf, Leaf> Leaf::MakeChilds(uint32_t feature_number, bin_id bin_number, float_type left_weight, float_type right_weight) const {
    std::vector<uint32_t> left_rows, right_rows;
    for(uint32_t index : row_indexes_) {
        if(dataset_->GetFeature(index, feature_number) <= bin_number) {
            left_rows.push_back(index);
        } else {
            right_rows.push_back(index);
        }
    }
    Leaf left(leaf_index_ * 2 + 1, left_weight, left_rows, dataset_, optData_);
    Leaf right(leaf_index_ * 2 + 2, right_weight, right_rows, dataset_, optData_);
    return std::make_tuple(left, right);
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

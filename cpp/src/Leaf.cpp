#include "Leaf.h"
#include <cmath>


float_type Leaf::CalculateGain(float_type gradient,
                               float_type hessian) const {
    float_type denominator = hessian + lambda_l2_reg_;
    float_type gain = 0;
    if(denominator != 0) {
        gain = -pow(gradient, 2.0) / (2.0 * denominator);
    }
    return gain;
}

float_type Leaf::CalculateWeight(float_type gradient,
                                 float_type hessian) const {
    float_type denominator = hessian + lambda_l2_reg_;
    float_type weight = 0;
    if(denominator != 0) {
        weight = -gradient / denominator;
    }
    return weight;
}

Histogram Histogram::operator-(const Histogram& other) const {
    Histogram new_hist;
    /*for(uint32_t i = 0; i < gradient_cs_.size(); ++i) {
        new_hist.gradient_cs_[i] -= other.gradient_cs_[i];
        new_hist.hessian_cs_[i] -= other.hessian_cs_[i];
    }*/
    return new_hist;
}


/*void Leaf::CalculateHistogram(uint32_t feature_number,
                             float_type lambda_l2_reg,
                             uint32_t bin_count,
                             const std::vector<bin_id>& feature_vector,
                             const std::vector<float_type>& gradients,
                             const std::vector<float_type>& hessians) {
    if(row_indices_.empty()) {
        return;
    }
    std::vector<float_type> gradient_sum(bin_count, 0);
        std::vector<float_type> hessian_sum(bin_count, 0);
    for(uint32_t index : row_indices_) {
        bin_id bin_number = feature_vector[index];
        gradient_sum[bin_number] += gradients[index];
        hessian_sum[bin_number] += hessians[index];
    }
    for(uint32_t i = 1; i < bin_count; ++i) {
        gradient_sum[i] += gradient_sum[i - 1];
        hessian_sum[i] += hessian_sum[i - 1];
    }
    histograms_[feature_number] = 
        Histogram(gradient_sum, hessian_sum, lambda_l2_reg, false, bin_count);
}*/

void Leaf::DiffHistogram(uint32_t feature_number, const Leaf& parent,
                              const Leaf& sibling) {
    histograms_[feature_number] = parent.histograms_[feature_number] -
            sibling.histograms_[feature_number];
}

void Leaf::CopyHistogram(uint32_t feature_number, const Leaf& parent) {
    histograms_[feature_number] = parent.histograms_[feature_number];
}

Leaf Leaf::MakeChild(bool is_left,
                     Dataset* dataset,
                     uint32_t feature_number,
                     bin_id bin_number,
                     float_type weight,
                     uint32_t depth) const {
    std::vector<uint32_t> child_rows;
    auto data = dataset->GetData();
    uint32_t child_index;
    if(is_left) {
        child_index = leaf_index_ * 2 + 1;
    } else {
        child_index = leaf_index_ * 2 + 2;
    }

    for(uint32_t index : row_indices_) {
        bool belongs_left = data[index].bin_ids[feature_number] <= bin_number;
        if(belongs_left == is_left) {
            child_rows.push_back(index);
            data[index].leaf_index = (child_index + 1)  - uint32_t(pow(2, depth + 1));
        }
    }

    uint32_t child_hist_size = histogram_count_;
    if(child_rows.empty()) {
        weight = weight_;
        child_hist_size = 0;
    }

    Leaf child(child_index, weight, child_hist_size, child_rows, bin_counts_, lambda_l2_reg_);
    return child;
}

uint32_t Leaf::GetIndex(uint32_t depth) const {
    return (leaf_index_ + 1)  - uint32_t(pow(2, depth));
}

uint32_t Leaf::ParentVectorIndex(uint32_t base) const {
    uint32_t parent_index = (leaf_index_ - 1) / 2;
    return (parent_index + 1) - base;
}
    

bool Leaf::IsEmpty() const {
    return row_indices_.empty();
}

float_type Leaf::GetWeight() const {
    return weight_;
}


float_type Leaf::CalculateSplitGain(uint32_t feature_number, 
                                    bin_id bin_number) const {
    if(row_indices_.empty()) {
        return 0;
    }

    float_type gain_left = CalculateGain(
            histograms_[feature_number].gradients_hessians[bin_number].gradient,
            histograms_[feature_number].gradients_hessians[bin_number].hessian);

    auto bin_count = bin_counts_[feature_number];

    float_type right_gradient_sum =
            histograms_[feature_number].gradients_hessians[bin_count - 1].gradient -
            histograms_[feature_number].gradients_hessians[bin_number].gradient;
    float_type right_hessian_sum =
            histograms_[feature_number].gradients_hessians[bin_count - 1].hessian -
            histograms_[feature_number].gradients_hessians[bin_number].hessian;
    float_type gain_right = CalculateGain(right_gradient_sum,
                                          right_hessian_sum);

    return gain_left + gain_right;
}

std::tuple<float_type, float_type> Leaf::CalculateSplitWeights(uint32_t feature_number,
                                                               bin_id bin_number) const {
    if(row_indices_.empty()) {
        return std::make_tuple(0.f, 0.f);
    }

    float_type weight_left = CalculateWeight(
            histograms_[feature_number].gradients_hessians[bin_number].gradient,
            histograms_[feature_number].gradients_hessians[bin_number].hessian);

    auto bin_count = bin_counts_[feature_number];

    float_type right_gradient_sum =
            histograms_[feature_number].gradients_hessians[bin_count].gradient -
            histograms_[feature_number].gradients_hessians[bin_number].gradient;
    float_type right_hessian_sum =
            histograms_[feature_number].gradients_hessians[bin_count].hessian -
            histograms_[feature_number].gradients_hessians[bin_number].hessian;
    float_type weight_right = CalculateWeight(right_gradient_sum,
                                              right_hessian_sum);

    return std::make_tuple(weight_left, weight_right);
}

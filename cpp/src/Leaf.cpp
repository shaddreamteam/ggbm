#include "Leaf.h"
#include <cmath>

void Histogram::AddGrad(bin_id bin_number, float_type row_gradient, 
                 float_type row_hessian) {
    gradient_[bin_number] += row_gradient;
    hessian_[bin_number] += row_hessian;
    row_count_[bin_number] += 1;
}

void Histogram::MakeCumsum() {
    for(uint32_t i = 1; i < n_bins; ++i) {
        gradient_[i] += gradient_[i - 1];
        hessian_[i] += hessian_[i - 1];
        row_count_[i] += row_count_[i - 1];
    }
}

std::tuple<std::vector<float_type>, std::vector<float_type>> Histogram::CalculateGain() {
    if(!finalized) {
        MakeCumsum();
        finalized = true;
    }
    std::vector<float_type> gain(n_bins, 0);
    std::vector<float_type> weight(n_bins, 0);
    for(i = 0; i < n_bins; ++i) {
        weight[i] = -gradient_[i] / (hessian_[i] + row_count_[i] * lambda_l2_reg);
        gain[i] = -pow(gradient_[i], 2.0) / 2.0 / (hessian_[i] + row_count_[i] * lambda_l2_reg);
    }
    return std::maketuple(gain, weight);
}

//float_type SubLeaf::CalculateGain() {
//    if (!row_count_) {
//        return 0.0;
//    }
//
//    weight_ = -gradient_ / (hessian_ + row_count_ * lambda_l2_reg_);
//    gain_ = -pow(gradient_, 2.0) / (hessian_ + row_count_ * lambda_l2_reg_) / 2.0;
//
//    return gain_;
//}

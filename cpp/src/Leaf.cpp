#include "Leaf.h"
#include <cmath>

void SubLeaf::AddRow(float_type row_gradient, float_type row_hessian) {
    gradient_ += row_gradient;
    hessian_ += row_hessian;
    ++row_count_;
}

float_type SubLeaf::CalculateGain() {
    if (!row_count_) {
        return 0.0;
    }

    weight_ = -gradient_ / (hessian_ + row_count_ * lambda_l2_reg_);
    gain_ = -pow(gradient_, 2.0) / (hessian_ + row_count_ * lambda_l2_reg_) / 2.0;

    return gain_;
}
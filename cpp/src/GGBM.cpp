#include "GGBM.h"

float_type OptData::GetGradient(uint32_t row_number) const {
    return gradients.at(row_number);
}

float_type OptData::GetHessian(uint32_t row_number) const {
    return hessians.at(row_number);
}

float_type OptData::GetPrediction(uint32_t row_number) const {
    return predictions.at(row_number);
}

void OptData::SetPrediction(uint32_t row_number, float_type prediction) {
    predictions[row_number] = prediction;
}

#include <cmath>
#include <exception>
#include "Loss.h"

void MSE::SetGradientsAndHessians(Dataset* dataset) {
    auto data = dataset->GetData();
    auto size = dataset->GetRowCount();
    float_type hessian = 2.0 / size;
    for(uint32_t i = 0; i < size; ++i) {
        data[i].gradient = 2 * (data[i].prediction - data[i].target) / size;
        data[i].hessian = hessian;
    }
}

float_type MeanTarget(Dataset* dataset) {
    float_type mean = 0.;
    auto data = dataset->GetData();
    for(uint32_t i = 0; i < dataset->GetRowCount(); ++i) {
        mean += data[i].target;
    }
    return mean / dataset->GetRowCount();
}

void MSE::SetFirstPrediction(Dataset* dataset) {
    auto data = dataset->GetData();
    auto prediction = MeanTarget(dataset);
    dataset->SetBasePrediction(prediction);
}

float_type MSE::GetLoss(Dataset* dataset) const {
    float_type loss = 0.;
    auto data = dataset->GetData();
    for(uint32_t i = 0; i < dataset->GetRowCount(); ++i) {
        loss += std::pow(data[i].prediction - data[i].target, 2.0);
    }
    loss /= dataset->GetRowCount();
    return loss;
}

float_type Sigmoid(float_type logit) {
    return 1. / (1. + std::exp(-logit));
}

void LogLoss::SetGradientsAndHessians(Dataset* dataset) {
    auto data = dataset->GetData();
    auto size = dataset->GetRowCount();
    for(uint32_t i = 0; i < size; ++i) {
        float_type probability = Sigmoid(data[i].prediction);
        data[i].gradient = -(data[i].target - probability) / size;
        data[i].hessian = probability * (1 - probability) / size;
    }
}


void LogLoss::SetFirstPrediction(Dataset* dataset) {
    auto data = dataset->GetData();
    auto prediction = MeanTarget(dataset);
    dataset->SetBasePrediction(prediction);
}

float_type LogLoss::GetLoss(Dataset* dataset) const {
    auto data = dataset->GetData();
    auto size = dataset->GetRowCount();
    float_type loss = 0.;
    for(uint32_t i = 0; i < size; ++i) {
        float_type probability = Sigmoid(data[i].prediction);
        float_type target = data[i].target;
        loss -= target * log(probability) + (1. - target) * log(1. - probability);
    }
    loss /= size;
    return loss;
}


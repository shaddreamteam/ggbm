#include <cmath>
#include <exception>
#include "Loss.h"

void Loss::check_correct_input(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    if (dataset.GetRowCount() != predictions.size()) {
        throw std::runtime_error("Target size doesn't match prediction size");
    }
    if (dataset.GetRowCount() == 0) {
        throw std::runtime_error("No targets specified");
    }
}

std::vector<float_type> MSE::GetGradients(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> gradients(dataset.GetRowCount());
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        gradients[i] = 2 * (predictions.at(i) - dataset.GetTarget(i)) /
            dataset.GetRowCount();
    }
    return gradients;
}

std::vector<float_type> MSE::GetHessians(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> hessians(dataset.GetRowCount(), 2.0 /
            dataset.GetRowCount());
    return hessians;
}

float_type MeanTarget(const TrainDataset& dataset) {
    float_type mean = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        mean += dataset.GetTarget(i) / dataset.GetRowCount();
    }
    return mean;
}

float_type MSE::GetFirstPrediction(const TrainDataset& dataset) const{
    return MeanTarget(dataset);
}

float_type MSE::GetLoss(const TrainDataset& dataset, 
                        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    float_type loss = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        loss += (predictions[i] - dataset.GetTarget(i)) *
            (predictions[i] - dataset.GetTarget(i));
    }
    loss /= dataset.GetRowCount();
    return loss;
}

float_type Sigmoid(float_type logit) {
    return 1 / (1 + std::exp(-logit));
}

std::vector<float_type> LogLoss::GetGradients(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> gradients(dataset.GetRowCount());
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        gradients[i] = -(dataset.GetTarget(i) - Sigmoid(predictions[i])) /
            dataset.GetRowCount();
    }
    return gradients;
}

std::vector<float_type> LogLoss::GetHessians(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> hessians(dataset.GetRowCount());
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        float_type probability = Sigmoid(predictions[i]);
        hessians[i] = probability * (1 - probability) / dataset.GetRowCount();
    }
    return hessians;
}

float_type LogLoss::GetFirstPrediction(const TrainDataset& dataset) const{
    return MeanTarget(dataset);
}

float_type LogLoss::GetLoss(const TrainDataset& dataset, 
                        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    float_type loss = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        float_type probability = Sigmoid(predictions[i]);
        float_type target = dataset.GetTarget(i);
        loss -= target * log(probability) + 
            (1 - target) * log(1 - probability);
    }
    loss /= dataset.GetRowCount();
    return loss;
}


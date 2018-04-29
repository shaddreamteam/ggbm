#include <exception>
#include "Loss.h"

void Loss::check_correct_input(const TrainDataset& dataset, const std::vector<float_type>& predictions) const {
    if (dataset.GetNRows() != predictions.size()) {
        throw std::runtime_error("Target size doesn't match prediction size");
    }
    if (dataset.GetNRows() == 0) {
        throw std::runtime_error("No targets specified");
    }
}

std::vector<float_type> MSE::GetGradients(const TrainDataset& dataset, const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> gradients(dataset.GetNRows());
    for(uint32_t i = 0; i < dataset.GetNRows(); ++i) {
        gradients[i] = 2 * (predictions.at(i) - dataset.GetTarget(i)) / dataset.GetNRows();
    }
    return gradients;
}

std::vector<float_type> MSE::GetHessians(const TrainDataset& dataset, const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    std::vector<float_type> hessians(dataset.GetNRows(), 2.0 / dataset.GetNRows());
    return hessians;
}

float_type MeanTarget(const TrainDataset& dataset) {
    float_type mean = 0;
    for(uint32_t i = 0; i < dataset.GetNRows(); ++i) {
        mean += dataset.GetTarget(i) / dataset.GetNRows();
    }
    return mean;
}

float_type MSE::GetFirstPrediction(const TrainDataset& dataset) const{
    return MeanTarget(dataset);
}

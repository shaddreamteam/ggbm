#include "OptData.h"

OptData::OptData(const TrainDataset& dataset,
                 const std::vector<float_type>& predictions,
                 const Loss& loss): 
        predictions_(predictions) {
    hessians_ = loss.GetHessians(dataset, predictions);
    gradients_ = loss.GetGradients(dataset, predictions);
}

float_type OptData::GetGradient(uint32_t row_number) const {
    return gradients_.at(row_number);
}

float_type OptData::GetHessian(uint32_t row_number) const {
    return hessians_.at(row_number);
}

float_type OptData::GetPrediction(uint32_t row_number) const {
    return predictions_.at(row_number);
}

void OptData::Update(const TrainDataset& dataset,
            const std::vector<float_type>& increment_predictions,
            float_type learning_rate,
            const Loss& loss) {
    for(uint32_t i = 0; i < predictions_.size(); ++i) {
       predictions_[i] += increment_predictions[i] * learning_rate; 
    }
    hessians_ = loss.GetHessians(dataset, predictions_);
    gradients_ = loss.GetGradients(dataset, predictions_);
}

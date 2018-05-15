/*
#include "OptData.h"

OptData::OptData(const TrainDataset& dataset,
                 const std::vector<float_type>& predictions,
                 const Loss& loss): 
        predictions_(predictions) {
    hessians_ = loss.SetHessians(dataset);
    gradients_ = loss.SetGradientsAndHessians(dataset);
}

const std::vector<float_type>& OptData::GetGradients() const {
    return gradients_;
}

const std::vector<float_type>& OptData::GetHessians() const {
    return hessians_;
}

float_type OptData::GetPrediction(uint32_t row_number) const {
    return predictions_.at(row_number);
}

void OptData::Update(const TrainDataset& dataset,
                     const std::vector<float_type>& increment_predictions,
                     const Loss& loss) {
    for(uint32_t i = 0; i < predictions_.size(); ++i) {
       predictions_[i] += increment_predictions[i];
    }
    hessians_ = loss.SetHessians(dataset);
    gradients_ = loss.SetGradientsAndHessians(dataset);
}
*/

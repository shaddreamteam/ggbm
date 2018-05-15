#include "OptData.h"

OptData::OptData(const TrainDataset& dataset,
                 const std::vector<float_type>& predictions,
                 Loss& loss) {
    Update(dataset, predictions, loss);
}

const std::vector<float_type>& OptData::GetGradients() const {
    return gradients_;
}

const std::vector<float_type>& OptData::GetHessians() const {
    return hessians_;
}

void OptData::Update(const TrainDataset& dataset,
                     const std::vector<float_type>& predictions,
                     Loss& loss) {
    loss.UpdateGradientsAndHessians(dataset, predictions, &gradients_, &hessians_);
}

#ifndef LOSS_H 
#define LOSS_H

#include <vector>
#include "Base.h"
#include "Dataset.h"

class Loss {
public:
    virtual std::vector<float_type> GetGradients(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const = 0;

    virtual std::vector<float_type> GetHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const = 0;

    virtual float_type GetFirstPrediction(
            const TrainDataset& dataset) const = 0;

    virtual float_type GetLoss(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const = 0;

protected:
    void check_correct_input(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;
};

class MSE : public Loss {
public:
    MSE() {};
    std::vector<float_type> GetGradients(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;

    std::vector<float_type> GetHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;

    float_type GetFirstPrediction(const TrainDataset& dataset) const;

    float_type GetLoss(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;
};

float_type Sigmoid(float_type logit);

class LogLoss : public Loss {
public:
    LogLoss() {};
    std::vector<float_type> GetGradients(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;

    std::vector<float_type> GetHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;

    float_type GetFirstPrediction(const TrainDataset& dataset) const;

    float_type GetLoss(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;
};
#endif //LOSS_H

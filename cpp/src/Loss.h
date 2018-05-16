#ifndef LOSS_H 
#define LOSS_H

#include <vector>
#include "Base.h"
#include "Dataset.h"

class Loss {
public:
    virtual void UpdateGradientsAndHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions,
            std::vector<float_type>* gradients,
            std::vector<float_type>* hessians) = 0;

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
    MSE(const Config& config) : config_(config) {};
    void UpdateGradientsAndHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions,
            std::vector<float_type>* gradients,
            std::vector<float_type>* hessians);

    float_type GetFirstPrediction(const TrainDataset& dataset) const;

    float_type GetLoss(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;
private:
    const Config& config_;
};

float_type Sigmoid(float_type logit);

class LogLoss : public Loss {
public:
    LogLoss(const Config& config) : config_(config) {};
    void UpdateGradientsAndHessians(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions,
            std::vector<float_type>* gradients,
            std::vector<float_type>* hessians);

    float_type GetFirstPrediction(const TrainDataset& dataset) const;

    float_type GetLoss(
            const TrainDataset& dataset,
            const std::vector<float_type>& predictions) const;
private:
    const Config& config_;
};
#endif //LOSS_H

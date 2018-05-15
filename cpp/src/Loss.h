#ifndef LOSS_H 
#define LOSS_H

#include <vector>
#include "Base.h"
#include "Dataset.h"

class Loss {
public:
    virtual void SetGradientsAndHessians(Dataset* dataset) = 0;

    virtual void SetFirstPrediction(Dataset* dataset) = 0;

    virtual float_type GetLoss(Dataset* dataset) const = 0;
};

class MSE : public Loss {
public:
    MSE() {};
    void SetGradientsAndHessians(Dataset* dataset);

    void SetFirstPrediction(Dataset* dataset);

    float_type GetLoss(Dataset* dataset) const;
};

float_type Sigmoid(float_type logit);

class LogLoss : public Loss {
public:
    LogLoss() {};
    void SetGradientsAndHessians(Dataset* dataset);

    void SetFirstPrediction(Dataset* dataset);

    float_type GetLoss(Dataset* dataset) const;
};
#endif //LOSS_H

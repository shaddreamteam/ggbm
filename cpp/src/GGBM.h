#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <memory>
#include <vector>
#include <string>
#include "Base.h"
#include "Loss.h"
#include "OptData.h"
#include "Tree.h"

enum ObjectiveType {
    kMse,
    kLogLoss,
};

class GGBM {
public:
    GGBM(uint32_t thread_count, ObjectiveType objective): 
        thread_count_(thread_count), objective_(objective) {};
    void Train(
            const TrainDataset& trainData,
            const Loss& loss,
            uint32_t depth,
            uint32_t n_estimators,
            float_type lambda_l2_reg,
            float_type learning_rate, 
            float_type row_sampling,
            uint32_t min_subsample);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const; 
private:
    uint32_t thread_count_;
    float_type base_prediction_;
    float_type learning_rate_;
    std::vector<Tree> trees_;
    ObjectiveType objective_;
};
#endif //CPP_GGBM_H

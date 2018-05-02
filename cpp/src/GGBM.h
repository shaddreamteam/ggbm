#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <memory>
#include <vector>
#include <string>
#include "Base.h"
#include "Loss.h"
#include "OptData.h"
#include "Tree.h"


class GGBM {
public:
    GGBM(): initialized_(false) {};
    void Train(const std::shared_ptr<const TrainDataset> trainData, const Loss& loss, 
            uint32_t depth, uint32_t n_estimators, float_type lambda_l2_reg, 
            float_type learning_rate, float_type row_sampling, 
            uint32_t min_subsample);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const; 
private:
    bool initialized_;
    float_type base_prediction_;
    float_type learning_rate_;
    std::vector<Tree> trees_;
};
#endif //CPP_GGBM_H

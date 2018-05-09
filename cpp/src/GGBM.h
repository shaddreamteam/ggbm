#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <memory>
#include <vector>
#include <string>
#include "Base.h"
#include "Loss.h"
#include "OptData.h"
#include "Tree.h"
#include "Config.h"
#include "FeatureTransformer.h"


class GGBM {
public:
    GGBM(Config& config, FeatureTransformer feature_transformer): 
        thread_count_(config.GetThreads()),
        objective_(config.GetObjective()),
        feature_transformer_(feature_transformer) {};

    void Train(const Config& config, const TrainDataset& trainDataset,
               const Loss& loss);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const; 

private:
    uint32_t thread_count_;
    float_type base_prediction_;
    float_type learning_rate_;
    std::vector<Tree> trees_;
    ObjectiveType objective_;
    FeatureTransformer feature_transformer_;
};
#endif //CPP_GGBM_H

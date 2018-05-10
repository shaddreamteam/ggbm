#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include "Base.h"
#include "Loss.h"
#include "OptData.h"
#include "Tree.h"
#include "Config.h"
#include "FeatureTransformer.h"


class GGBM {
public:
    GGBM(const Config& config) :
            config_(config),
            objective_(config.GetObjective()),
            feature_transformer_(std::make_shared<FeatureTransformer>(config)) {};

    void Train(const TrainDataset& trainDataset);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const;

    void Save(std::ofstream& stream);
    void Load(std::ifstream& stream);

    std::shared_ptr<FeatureTransformer> GetFeatureTransformer();

private:
    float_type base_prediction_;
    std::vector<Tree> trees_;
    const Config& config_;
    ObjectiveType objective_;
    std::shared_ptr<FeatureTransformer> feature_transformer_;
};
#endif //CPP_GGBM_H

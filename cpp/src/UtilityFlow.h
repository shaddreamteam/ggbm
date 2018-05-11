#ifndef CPP_UTILITYFLOW_H
#define CPP_UTILITYFLOW_H


#include <unordered_map>

#include "GGBM.h"
#include "InputParser.h"


class UtilityFlow{
public:
    UtilityFlow() {};    

    void Start(int argc, char **argv) {
        Config cfg;
        InputParser parser;
        parser.ParseArgs(argc, argv, &cfg);
        
        GGBM ggbm(cfg);
        if(cfg.GetMode() == "train") {
            TrainModel(cfg, &ggbm);
            // predict for testing:
            MakePrediction(cfg, &ggbm);
        } else if(cfg.GetMode() == "predict") {
            LoadModel(cfg, &ggbm);
            MakePrediction(cfg, &ggbm);
        }      
    }

private:

    void LoadModel(Config& cfg, GGBM* ggbm) {
        std::ifstream in_model_file(cfg.GetModelFilename());
        ggbm->Load(in_model_file);
    }

    void TrainModel(Config& cfg, GGBM* ggbm) {
        TrainDataset dataset(cfg.GetTrainFilename(), ggbm->GetFeatureTransformer());
        ggbm->Train(dataset);
        std::ofstream out_model_file(cfg.GetModelFilename());
        ggbm->Save(out_model_file);
    }

    void MakePrediction(Config& cfg, GGBM* ggbm) {
        TestDataset test(cfg.GetTestFilename(), ggbm->GetFeatureTransformer(), cfg.GetFileHasTarget());
        auto preds = ggbm->PredictFromDataset(test);
        // save predictions
        for(int i = 0; i < preds.size() && i < 100; ++i) {
            std::cout << preds[i] << ' ';
        }
    }
};



#endif //CPP_UTILITYFLOW_H

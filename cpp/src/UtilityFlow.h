#ifndef CPP_UTILITYFLOW_H
#define CPP_UTILITYFLOW_H


#include <unordered_map>
#include <fstream>
#include <memory>

#include "GGBM.h"
#include "InputParser.h"


class UtilityFlow{
public:
    UtilityFlow() {};    

    void Start(int argc, char **argv) {
        InputParser parser;
        parser.ParseArgs(argc, argv);
        Config cfg(parser.config);
        
        GGBM ggbm(cfg);
        if(cfg.GetModelFilename().size() == 0) {
            TrainDataset dataset(cfg.GetTrainFilename(), ggbm.GetFeatureTransformer());
            ggbm.Train(dataset);
            std::ofstream out_model_file("model.bst");
            ggbm.Save(out_model_file);
        } else {
            std::ifstream in_model_file(cfg.GetModelFilename());
            ggbm.Load(in_model_file);
        }

        PredictionFlow(cfg, &ggbm);
    }

private:
    void PredictionFlow(Config& cfg,
                        GGBM* ggbm) {
        TestDataset test(cfg.GetTestFilename(), ggbm->GetFeatureTransformer(), false);
        auto preds = ggbm->PredictFromDataset(test);
        // save predictions
        for(int i = 0; i < preds.size() && i < 100; ++i) {
            std::cout << preds[i] << ' ';
        }
    }
};



#endif //CPP_UTILITYFLOW_H

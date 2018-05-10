#ifndef CPP_UTILITYFLOW_H
#define CPP_UTILITYFLOW_H


#include <unordered_map>

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
<<<<<<< HEAD
    void PredictionFlow(Config& cfg, const FeatureTransformer& ft,
                        GGBM& boosting) {
        TestDataset test(cfg.GetTestFilename(), ft, false);
        auto preds = boosting.PredictFromDataset(test);
=======
    void PredictionFlow(Config& cfg,
                        GGBM* ggbm) {
        TestDataset test(cfg.GetTestFilename(), ggbm->GetFeatureTransformer(), false);
        auto preds = ggbm->PredictFromDataset(test);
>>>>>>> ebb432260c01b96f3747bad7414aca6954b73fd4
        // save predictions
        for(int i = 0; i < preds.size() && i < 100; ++i) {
            std::cout << preds[i] << ' ';
        }
    }
};



#endif //CPP_UTILITYFLOW_H

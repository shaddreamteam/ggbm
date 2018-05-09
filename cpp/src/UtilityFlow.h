#ifndef CPP_UTILITYFLOW_H
#define CPP_UTILITYFLOW_H


#include <unordered_map>

#include "GGBM.h"
#include "InputParser.h"


class UtilityFlow{
public:
    UtilityFlow() {};    

    void Start(int argc, char **argv){
        InputParser parser(argc, argv);
        Config cfg(parser.config);

        if(cfg.GetModelFilename().size() == 0) {
            FeatureTransformer ft(cfg.GetThreads());
            TrainDataset dataset(cfg.GetTrainFilename(), ft);
            GGBM ggbm(cfg, ft);
            MSE loss;
            ggbm.Train(cfg, dataset, loss);
            // save model
            PredictionFlow(cfg, ft, ggbm);
        } else {
            // load model
            // PredictionFlow(cfg, ft);
        }


    }
private:
    void PredictionFlow(Config& cfg, const FeatureTransformer& ft,
                        GGBM& boosting) {
        TestDataset test(cfg.GetTestFilename(), ft, false);
        auto preds = boosting.PredictFromDataset(test);
        // save predictions
        for(int i = 0; i < preds.size() && i < 100; ++i) {
            std::cout << preds[i] << ' ';
        }
    }
};



#endif //CPP_UTILITYFLOW_H

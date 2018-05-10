#ifndef CPP_UTILITYFLOW_H
#define CPP_UTILITYFLOW_H


#include <unordered_map>
#include <fstream>

#include "GGBM.h"
#include "InputParser.h"


class UtilityFlow{
public:
    UtilityFlow() {};    

    void Start(int argc, char **argv){
        InputParser parser(argc, argv);
        Config cfg(parser.config);

        FeatureTransformer ft(cfg.GetThreads());
        GGBM ggbm(cfg, ft);
        if(cfg.GetModelFilename().size() == 0) {
            TrainDataset dataset(cfg.GetTrainFilename(), ft);
            MSE loss;
            ggbm.Train(cfg, dataset, loss);
            std::ofstream out_model_file("model.bst");
            ft.Save(out_model_file);
            ggbm.Save(out_model_file);
            // save model
            PredictionFlow(cfg, ft, ggbm);
        } else {
            std::ifstream in_model_file(cfg.GetModelFilename());
            ft.Load(in_model_file);
            ggbm.Load(in_model_file);
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

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
            if (!cfg.GetTestFilename().empty()) {
                MakePrediction(cfg, &ggbm);
            }
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
        CSVReader cr;
        std::vector<std::vector<float_type>> feature_values;
        std::vector<float_type> targets;
        cr.GetDataFromFile(cfg.GetTrainFilename(), &feature_values, true, &targets);
        auto ft = ggbm->GetFeatureTransformer();
        auto row_bin_ids = ft->FitTransform(feature_values);
        feature_values.clear();
        Dataset train_dataset(row_bin_ids, ft->GetBinCounts(), &targets);
        ggbm->Train(&train_dataset);
        std::ofstream out_model_file(cfg.GetModelFilename());
        ggbm->Save(out_model_file);
    }

    void MakePrediction(Config& cfg, GGBM* ggbm) {
        CSVReader cr;
        std::vector<std::vector<float_type>> feature_values;
        std::vector<float_type> targets;
        cr.GetDataFromFile(cfg.GetTestFilename(),
                           &feature_values,
                           cfg.GetFileHasTarget(),
                           &targets);
        auto ft = ggbm->GetFeatureTransformer();
        auto row_bin_ids = ft->FitTransform(feature_values);
        feature_values.clear();
        Dataset test_dataset(row_bin_ids, ft->GetBinCounts(), nullptr);

        auto preds = ggbm->PredictFromDataset(&test_dataset);

        std::ofstream stream(cfg.GetOutputFilename());
        for(int i = 0; i < preds.size() - 1; ++i) {
            stream << preds[i] << '\n';
        }
        stream << preds[preds.size() - 1];
    }
};



#endif //CPP_UTILITYFLOW_H

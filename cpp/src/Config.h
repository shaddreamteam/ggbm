#ifndef CPP_CONFIG_H
#define CPP_CONFIG_H

#include <unordered_map>


class Config{
    public:

        friend class InputParser;

        Config() {};
        Config(const Config&) = default;
        
        const std::string GetModelFilename() { return params_["filename_model"]; }
        const std::string GetTrainFilename() { return params_["filename_train"]; }
        const std::string GetTestFilename() { return params_["filename_test"]; }
        const uint32_t GetThreads() { return std::stoi(params_["threads"]); }
        const std::string GetLoss() { return params_["loss"]; }
        const ObjectiveType GetObjective() { return obj_dict_[params_["objective"]]; }
        const float_type GetLearningRate() { return std::stod(params_["learning_rate"]); }
        const float_type GetNEstimators() { return std::stod(params_["n_estimators"]); }
        const uint32_t GetDepth() { return std::stoi(params_["depth"]); }
        const float_type GetLambdaL2() { return std::stod(params_["lambda"]); }
        const float_type GetRowSampling() { return std::stod(params_["row_sampling"]); }
        const uint32_t GetMinSubsample() { return std::stoi(params_["min_subsample"]); }
    private:
        std::unordered_map<std::string, std::string> params_={
            {"filename_model", "\0"},
            {"filename_train", "\0"},
            {"filename_test", "\0"},
            {"threads", "1"},
            {"loss", "mse"},
            {"objective", "mse"},
            {"learning_rate", "0.1"},
            {"n_estimators", "10"},
            {"depth", "5"},
            {"lambda", "1"},
            {"row_sampling", "1"},
            {"min_subsample", "1"},
        };

        std::unordered_map<std::string, ObjectiveType> obj_dict_={
            {"mse", kMse},
            {"logloss", kLogLoss}
        };
};

#endif //CPP_CONFIG_H
#ifndef CPP_CONFIG_H
#define CPP_CONFIG_H

#include <unordered_map>


class Config{
    public:

        friend class InputParser;

        Config() {};
        Config(const Config&) = default;
        
        const std::string GetModelFilename() const { return params_.at("filename_model"); }
        const std::string GetTrainFilename() const { return params_.at("filename_train"); }
        const std::string GetTestFilename() const { return params_.at("filename_test"); }
        const uint32_t GetThreads() const { return std::stoi(params_.at("threads")); }
        const std::string GetLoss() const { return params_.at("loss"); }
        const ObjectiveType GetObjective() const { return obj_dict_.at(params_.at("objective")); }
        const float_type GetLearningRate() const { return std::stod(params_.at("learning_rate")); }
        const float_type GetNEstimators() const { return std::stod(params_.at("n_estimators")); }
        const uint32_t GetDepth() const { return std::stoi(params_.at("depth")); }
        const float_type GetLambdaL2() const { return std::stod(params_.at("lambda")); }
        const float_type GetRowSampling() const { return std::stod(params_.at("row_sampling")); }
        const uint32_t GetMinSubsample() const { return std::stoi(params_.at("min_subsample")); }
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

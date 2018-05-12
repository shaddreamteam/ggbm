#ifndef CPP_CONFIG_H
#define CPP_CONFIG_H

#include "ConfigValue.h"


class Config{
public:
    friend class InputParser;

    Config() {};
    Config(const Config&) = default;

    const std::string GetMode() const {
        return params_.at("mode");
    }
    const std::string GetModelFilename() const {
        return params_.at("filename_model");
    }
    const std::string GetTrainFilename() const {
        return params_.at("filename_train");
    }
    const std::string GetTestFilename() const {
        return params_.at("filename_test");
    }
    const std::string GetOutputFilename() const {
        return params_.at("filename_output");
    }
    const uint32_t GetThreads() const {
        return params_.at("threads");
    }
    const ObjectiveType GetObjective() const {
        return params_.at("objective");
    }
    const float_type GetLearningRate() const {
        return params_.at("learning_rate");
    }
    const uint32_t GetNEstimators() const {
        return params_.at("n_estimators");
    }
    const uint32_t GetDepth() const {
        return params_.at("depth");
    }
    const float_type GetLambdaL2() const {
        return params_.at("lambda");
    }
    const float_type GetRowSampling() const {
        return params_.at("row_subsampling");
    }
    const uint32_t GetMinSubsample() const {
        return params_.at("min_subsample");
    }
    const bool GetFileHasTarget() const {
        return params_.at("file_has_target");
    }
private:
    std::unordered_map<std::string, ConfigValue> params_={
        {"mode", ConfigValue("", kString)},
        {"filename_model", ConfigValue("", kString)},
        {"filename_train", ConfigValue("", kString)},
        {"filename_test", ConfigValue("", kString)},
        {"filename_output", ConfigValue("", kString)},
        {"threads", ConfigValue("1", kUint32_t)},
        {"objective", ConfigValue("mse", kObjectiveType)},
        {"learning_rate", ConfigValue("0.1", kFloatType)},
        {"n_estimators", ConfigValue("10", kUint32_t)},
        {"depth", ConfigValue("5", kUint32_t)},
        {"lambda", ConfigValue("1", kFloatType)},
        {"row_subsampling", ConfigValue("1", kFloatType)},
        {"min_subsample", ConfigValue("1", kUint32_t)},
        {"file_has_target", ConfigValue("false", kBool)},
    };
};


#endif //CPP_CONFIG_H

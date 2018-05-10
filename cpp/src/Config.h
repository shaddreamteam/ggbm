#ifndef CPP_CONFIG_H
#define CPP_CONFIG_H

#include "ConfigValue.h"


class Config{
    public:
        friend class InputParser;

        Config() {};
        Config(const Config&) = default;
        
        const std::string GetModelFilename() const { 
            return params_.at("filename_model"); 
        }
        const std::string GetTrainFilename() const { 
            return params_.at("filename_train"); 
        }
        const std::string GetTestFilename() const { 
            return params_.at("filename_test"); 
        }
        const uint32_t GetThreads() const {
            return params_.at("threads"); 
        }
        const std::string GetLoss() const {
            return params_.at("loss"); 
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
    private:
        std::unordered_map<std::string, ConfigValue> params_={
            {"filename_model", ConfigValue("\0", kString)},
            {"filename_train", ConfigValue("\0", kString)},
            {"filename_test", ConfigValue("\0", kString)},
            {"threads", ConfigValue("1", kUint32_t)},
            {"loss", ConfigValue("mse", kString)},
            {"objective", ConfigValue("mse", kString)},
            {"learning_rate", ConfigValue("0.1", kFloatType)},
            {"n_estimators", ConfigValue("10", kUint32_t)},
            {"depth", ConfigValue("5", kUint32_t)},
            {"lambda", ConfigValue("1", kFloatType)},
            {"row_subsampling", ConfigValue("1", kFloatType)},
            {"min_subsample", ConfigValue("1", kUint32_t)},
        };
};


#endif //CPP_CONFIG_H

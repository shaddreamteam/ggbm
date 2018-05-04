#ifndef CPP_INPUTPARSER_H
#define CPP_INPUTPARSER_H


#include <string>
#include <unordered_map>
#include "Base.h"

#include <iostream>

class Config{
    public:
        friend class InputParser;

        Config() {};
        Config(const Config&) = default;
        
        const std::string GetFilename() { return params_["filename"]; }
        const float_type GetLearningRate() { return std::stod(params_["learning_rate"]); }
        const float_type GetNEstimators() { return std::stod(params_["n_estimators"]); }
        const float_type GetDepth() { return std::stod(params_["depth"]); }
        const float_type GetLambdaL2() { return std::stod(params_["lambda_l2_reg"]); }
        const std::string GetLoss() { return params_["loss"]; }
        const float_type GetRowSampling() { return std::stod(params_["row_sampling"]); }
        const float_type GetMinSubsample() { return std::stod(params_["min_subsample"]); }
        const uint32_t GetNThreads() { return std::stoi(params_["n_threads"]); }
    private:
        std::unordered_map<std::string, std::string> params_={
            {"filename", "\0"},
            {"learning_rate", "0.1"},
            {"n_estimators", "10"},
            {"depth", "5"},
            {"lambda_l2_reg", "1"},
            {"loss", "mse"},
            {"row_sampling", "1"},
            {"min_subsample", "1"},
            {"n_threads", "1"},
        };
};

class InputParser{
    // parse commands like ./cpp filename=train.csv learning_rate=0.2
    public:
        InputParser() {};
        // InputParser(int &argc, char **argv);
        InputParser(int &argc, char **argv){
            for (int i = 1; i < argc; ++i) {
                ParseToken(std::string(argv[i]));
            }
        };

        Config config;

    private:
        // void ParseToken(const std::string& token);
        void ParseToken(const std::string& token) {
            auto pos = token.find('=');
            config.params_[token.substr(0, pos)] = token.substr(pos+1);
        }

};

#endif //CPP_INPUTPARSER_H
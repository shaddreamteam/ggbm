#ifndef CPP_INPUTPARSER_H
#define CPP_INPUTPARSER_H

#include <string>
#include "Config.h"


class InputParser{
public:
    InputParser() {};
    void ParseArgs(int &argc, char **argv, Config* config){
        for (int i = 1; i < argc; ++i) {
            ParseToken(std::string(argv[i]), config);
        }

        std::string mode = config->params_.at("mode");
        if(mode == "\0") {
            throw std::invalid_argument("mode is not given");
        } else if(!(mode == "train" || mode == "predict")) {
            throw std::invalid_argument("Ivalid argument for mode");
        }
    };

    // Config config;

private:
    void ParseToken(const std::string& token, Config* config) {
        auto pos = token.find('=');
        std::string param_name = token.substr(0, pos);
        auto type = config->params_.at(param_name).GetType();
        try {
            config->params_.at(param_name) = ConfigValue(token.substr(pos+1), type);
        } catch(const std::invalid_argument& e) {
            throw std::invalid_argument("Ivalid argument for " + param_name);
        }
    }
};

#endif //CPP_INPUTPARSER_H

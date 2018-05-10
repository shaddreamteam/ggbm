#ifndef CPP_INPUTPARSER_H
#define CPP_INPUTPARSER_H

#include <string>

// #include "Base.h"
#include "Config.h"


class InputParser{
public:
    InputParser() {};
    void ParseArgs(int &argc, char **argv){
        for (int i = 1; i < argc; ++i) {
            ParseToken(std::string(argv[i]));
        }
    };

    Config config;

private:
    void ParseToken(const std::string& token) {
        auto pos = token.find('=');
        auto type = config.params_.at(token.substr(0, pos)).GetType();
        config.params_.at(token.substr(0, pos)) = ConfigValue(token.substr(pos+1), type);
    }
};

#endif //CPP_INPUTPARSER_H

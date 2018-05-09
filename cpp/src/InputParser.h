#ifndef CPP_INPUTPARSER_H
#define CPP_INPUTPARSER_H

#include <string>

#include "Base.h"
#include "Config.h"


class InputParser{
    public:
        InputParser() {};
        InputParser(int &argc, char **argv){
            for (int i = 1; i < argc; ++i) {
                ParseToken(std::string(argv[i]));
            }
        };

        Config config;

    private:
        void ParseToken(const std::string& token) {
            auto pos = token.find('=');
            config.params_[token.substr(0, pos)] = token.substr(pos+1);
        }
};

#endif //CPP_INPUTPARSER_H

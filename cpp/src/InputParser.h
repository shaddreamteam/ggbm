#ifndef CPP_INPUTPARSER_H
#define CPP_INPUTPARSER_H

#include <string>
#include "Config.h"


class InputParser{
public:
    InputParser() {};
    void ParseArgs(int &argc, char **argv, Config* config);

private:
    void ParseToken(const std::string& token, Config* config);
};

#endif //CPP_INPUTPARSER_H

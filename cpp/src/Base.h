#ifndef CPP_BASE_H
#define CPP_BASE_H

#define DEBUG

#include <iostream>


typedef float float_type;
typedef uint8_t bin_id;

const unsigned char kMaxBin = 255;
const float_type EPS = 1e-5;

enum ObjectiveType {
    kMse,
    kLogLoss,
};



#endif //CPP_BASE_H

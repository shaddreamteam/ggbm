#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <vector>
#include <string>
#include "Base.h"
class OptData {
public:
    float_type GetGradient(uint32_t row_number);    
    float_type GetHessian(uint32_t row_number);    
    float_type GetPrediction(uint32_t row_number);    
    void SetPrediction(uint32_t row_number, float_type prediction);

private:
    std::vector<float_type> gradients;
    std::vector<float_type> hessians;
    std::vector<float_type> predictions;
}
#endif //CPP_GGBM_H

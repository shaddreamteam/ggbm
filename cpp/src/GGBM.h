#ifndef CPP_GGBM_H
#define CPP_GGBM_H

#include <vector>
#include <string>
#include "Base.h"
class OptData {
public:
    OptData(std::vector<float_type> gradients, std::vector<float_type> hessians,
            std::vector<float_type> predictions): 
        gradients(gradients),
        hessians(hessians),
        predictions(predictions) {}

    float_type GetGradient(uint32_t row_number) const;
    float_type GetHessian(uint32_t row_number) const;    
    float_type GetPrediction(uint32_t row_number) const;    
    void SetPrediction(uint32_t row_number, float_type prediction);

private:
    std::vector<float_type> gradients;
    std::vector<float_type> hessians;
    std::vector<float_type> predictions;
};
#endif //CPP_GGBM_H

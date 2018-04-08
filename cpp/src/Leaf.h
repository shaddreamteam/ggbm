#ifndef CPP_LEAF_H
#define CPP_LEAF_H

#include <cstdint>
#include "Base.h"

class SubLeaf {
public:
    SubLeaf(float_type lambda_l2_reg) : gradient_(0.0),
                                        hessian_(0.0),
                                        gain_(0.0),
                                        lambda_l2_reg_(lambda_l2_reg),
                                        weight_(0.0),
                                        row_count_(0) {};

    void AddRow(float_type row_gradient, float_type row_hessian);

    float_type CalculateGain();

private:
    float_type gradient_;
    float_type hessian_;
    float_type gain_;
    float_type lambda_l2_reg_;
    float_type weight_;
    uint32_t row_count_;

};

class Leaf {
    Leaf();



};


#endif //CPP_LEAF_H

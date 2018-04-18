#ifndef CPP_LEAF_H
#define CPP_LEAF_H

#include <vector>
#include <tuple>
#include <cstdint>
#include "Base.h"
#include "GGBM.h"

class Leaf {
    Leaf();



};

class Histogram {
public:
    void Histogram(uint32_t n_bins, float_type lambda_l2_reg) : 
                                        gradient_(n_bins, 0.0),
                                        hessian_(n_bins, 0.0),
                                        gain_(n_bins, 0.0),
                                        lambda_l2_reg_(lambda_l2_reg),
                                        weight_(0.0),
                                        row_count_(n_bins, 0),
                                        finalized(false),
                                        n_bins(n_bins) {};

    void AddGrad(bin_id bin_number, float_type row_gradient, 
                 float_type row_hessian);

    std::tuple<std::vector<float_type>, std::vector<float_type>> CalculateGain();

private:
    std::vector<float_type> gradient_;
    std::vector<float_type> hessian_;
    float_type lambda_l2_reg_;
    std::vector<uint32_t> row_count_;
    bool finalized;
    uint32_t n_bins;

    void MakeCumsum();
}
#endif //CPP_LEAF_H

#ifndef CPP_LEAF_H
#define CPP_LEAF_H

#include <vector>
#include <tuple>
#include <cstdint>
#include "Base.h"
#include "GGBM.h"
#include "Dataset.h"

class Histogram {
public:
    Histogram(std::vector<float_type> gradient_cs, std::vector<float_type> hessian_cs,
              std::vector<uint32_t> row_count_cs, float_type lambda_l2_reg) :
        gradient_cs_(gradient_cs),
        hessian_cs_(hessian_cs),
        row_count_cs_(row_count_cs),
        lambda_l2_reg_(lambda_l2_reg) {};

    std::tuple<float_type, float_type> CalculateGains(bin_id bin_number) const;
    std::tuple<float_type, float_type> CalculateWeights(bin_id bin_number) const;

private:
    std::vector<float_type> gradient_cs_;
    std::vector<float_type> hessian_cs_;
    std::vector<uint32_t> row_count_cs_;
    float_type lambda_l2_reg_;

    float_type CalculateGain(float_type gradient, float_type hessian, 
                                         uint32_t row_count) const;
    float_type CalculateWeight(float_type gradient, float_type hessian, 
                                          uint32_t row_count) const;
};

class Leaf {
public:
    Leaf(std::vector<uint32_t> indexes, const Dataset& dataset, OptData& optData):
        indexes(indexes), dataset(dataset), optData(optData) {};

    Histogram GetWeightsGains(uint32_t feature_number);
private:
    float_type lambda_l2_reg_;
    std::vector<uint32_t> indexes;
    const Dataset& dataset;
    const OptData& optData;
};
#endif //CPP_LEAF_H

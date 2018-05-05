#ifndef CPP_LEAF_H
#define CPP_LEAF_H

#include <vector>
#include <tuple>
#include <cstdint>
#include <memory>
#include "Base.h"
#include "OptData.h"
#include "Dataset.h"

class Histogram {
public:
    Histogram(std::vector<float_type> gradient_cs, std::vector<float_type> hessian_cs,
              std::vector<uint32_t> row_count_cs, float_type lambda_l2_reg,
              bool empty_leaf) :
        gradient_cs_(gradient_cs),
        hessian_cs_(hessian_cs),
        row_count_cs_(row_count_cs),
        lambda_l2_reg_(lambda_l2_reg),
        empty_leaf_(empty_leaf) {};

    float_type CalculateSplitGain(bin_id bin_number) const;
    std::tuple<float_type, float_type> CalculateSplitWeights(bin_id bin_number) const;

private:
    std::vector<float_type> gradient_cs_;
    std::vector<float_type> hessian_cs_;
    std::vector<uint32_t> row_count_cs_;
    float_type lambda_l2_reg_;
    bool empty_leaf_;

    float_type CalculateGain(float_type gradient, float_type hessian, 
                                         uint32_t row_count) const;
    float_type CalculateWeight(float_type gradient, float_type hessian, 
                                          uint32_t row_count) const;
};

class Leaf {
public:
    Leaf() {};
    Leaf(uint32_t leaf_index, float_type weight, std::vector<uint32_t> row_indexes, std::shared_ptr<const TrainDataset> dataset, std::shared_ptr<const OptData> optData):
        leaf_index_(leaf_index), weight_(weight), row_indexes_(row_indexes), dataset_(dataset), optData_(optData) {};

    Histogram GetHistogram(uint32_t feature_number, float_type lambda_l2_reg) const;

    std::tuple<Leaf, Leaf> MakeChilds(uint32_t feature, bin_id bin_number, float_type left_weight, float_type right_weight) const;
 
    uint32_t GetIndex(uint32_t depth) const;
    bool IsEmpty() const;
    float_type GetWeight() const;
private:
    uint32_t leaf_index_;
    float_type weight_;
    std::vector<uint32_t> row_indexes_;
    std::shared_ptr<const TrainDataset> dataset_;
    std::shared_ptr<const OptData> optData_;
};
#endif //CPP_LEAF_H

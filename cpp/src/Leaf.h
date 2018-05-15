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
    Histogram() : gradient_cs_(0), hessian_cs_(0), lambda_l2_reg_(0),
                  empty_leaf_(true), bin_count_(0) {};

    Histogram(std::vector<float_type> gradient_cs,
              std::vector<float_type> hessian_cs,
              float_type lambda_l2_reg, bool empty_leaf,
              uint32_t bin_count) :
        gradient_cs_(gradient_cs), hessian_cs_(hessian_cs),
        lambda_l2_reg_(lambda_l2_reg), empty_leaf_(empty_leaf),
        bin_count_(bin_count) {};

    float_type CalculateSplitGain(bin_id bin_number) const;
    std::tuple<float_type, float_type> CalculateSplitWeights(bin_id bin_number) const;
    Histogram operator-(const Histogram& other) const;
private:
    std::vector<float_type> gradient_cs_;
    std::vector<float_type> hessian_cs_;
    float_type lambda_l2_reg_;
    bool empty_leaf_;
    uint32_t bin_count_;

    float_type CalculateGain(float_type gradient, float_type hessian) const;
    float_type CalculateWeight(float_type gradient, float_type hessian) const;
};

class Leaf {
public:
    Leaf() {};
    Leaf(uint32_t leaf_index, float_type weight, uint32_t n_features,
         std::vector<uint32_t> row_indexes):
        leaf_index_(leaf_index), weight_(weight), row_indices_(row_indexes),
        histograms_(n_features) {};

    void CalculateHistogram(uint32_t feature_number,
                            float_type lambda_l2_reg,
                            uint32_t bin_count,
                            const bin_id* feature_vector,
                            const std::vector<float_type>& gradients,
                            const std::vector<float_type>& hessians);
 
    void DiffHistogram(uint32_t feature_number, const Leaf& parent, 
                       const Leaf& sibling);

    void CopyHistogram(uint32_t feature_number, const Leaf& parent);

    float_type CalculateSplitGain(uint32_t feature_number,
                                 bin_id bin_number) const;

    std::tuple<float_type, float_type> CalculateSplitWeights(
            uint32_t feature_number, bin_id bin_number) const;

    Leaf MakeChild(bool is_left, const bin_id* feature_vector,
                   bin_id bin_number, float_type left_weight) const;
 
    uint32_t GetIndex(uint32_t depth) const;
    uint32_t ParentVectorIndex(uint32_t base) const;

    bool IsEmpty() const;
    float_type GetWeight() const;
    uint32_t Size() const { return row_indices_.size(); };

    const std::vector<uint32_t>& GetRowIndices();

    uint32_t leaf_index_;
private:
    float_type weight_;
    std::vector<uint32_t> row_indices_;
    std::vector<Histogram> histograms_;
};
#endif //CPP_LEAF_H

#ifndef CPP_LEAF_H
#define CPP_LEAF_H

#include <vector>
#include <tuple>
#include <cstdint>
#include <memory>
#include "Base.h"
#include "OptData.h"
#include "Dataset.h"

struct GradientAndHessian {
    float_type gradient;
    float_type hessian;
};

struct Histogram {
    Histogram operator-(const Histogram& other) const;
    void operator+=(const Histogram& other) {
        for (uint32_t bin = 0; bin < bin_count; ++bin) {
            gradients_hessians[bin].gradient += other.gradients_hessians[bin].gradient;
            gradients_hessians[bin].hessian += other.gradients_hessians[bin].hessian;
        }
    }

    void AddGradientAndHessian(bin_id bin, float_type gradient, float_type hessian) {
        gradients_hessians[bin].gradient += gradient;
        gradients_hessians[bin].hessian += hessian;
    }

    std::array<GradientAndHessian, kMaxBin> gradients_hessians;
    uint32_t bin_count;
};

class Leaf {
public:
    Leaf() = delete;
    Leaf(Leaf const&) = delete;
    Leaf& operator=(Leaf const&) = delete;

    Leaf(Leaf&& other) noexcept : bin_counts_(other.bin_counts_) {
        leaf_index_ = other.leaf_index_;
        weight_ = other.weight_;
        std::swap(row_indices_, other.row_indices_);
        histograms_ = other.histograms_;
        other.histograms_ = NULL;
        histogram_count_ = other.histogram_count_;
        lambda_l2_reg_ = other.lambda_l2_reg_;
    }

    Leaf& operator=(Leaf&& other) = delete;

    Leaf(uint32_t leaf_index,
         float_type weight,
         uint32_t histogram_count,
         std::vector<uint32_t> row_indices,
         const std::vector<uint32_t>& bin_counts,
         float_type lambda_l2_reg) : leaf_index_(leaf_index),
                                     weight_(weight),
                                     row_indices_(row_indices),
                                     bin_counts_(bin_counts),
                                     lambda_l2_reg_(lambda_l2_reg) {
        if (histogram_count) {
            histograms_ = (Histogram*) calloc(histogram_count, sizeof(Histogram));
        }
        for (uint32_t i = 0; i < histogram_count; ++i) {
            histograms_[i].bin_count = bin_counts[i % bin_counts_.size()];
        }
        histogram_count_ = histogram_count;
    };

    ~Leaf() {
        free(histograms_);
    }

    void CalculateHistogram(uint32_t feature_number,
                           float_type lambda_l2_reg,
                           uint32_t bin_count,
                           const std::vector<bin_id>& feature_vector,
                           const std::vector<float_type>& gradients,
                           const std::vector<float_type>& hessians);
 
    void DiffHistogram(uint32_t feature_number, const Leaf& parent, 
                       const Leaf& sibling);

    void CopyHistogram(uint32_t feature_number, const Leaf& parent);

    float_type CalculateSplitGain(uint32_t feature_number,
                                 bin_id bin_number) const;

    std::tuple<float_type, float_type> CalculateSplitWeights(
            uint32_t feature_number, bin_id bin_number) const;

    Leaf MakeChild(bool is_left,
                   Dataset* dataset,
                   uint32_t feature_number,
                   bin_id bin_number,
                   float_type left_weight,
                   uint32_t depth) const;
 
    uint32_t GetIndex(uint32_t depth) const;
    uint32_t ParentVectorIndex(uint32_t base) const;

    Histogram* GetHistograms() {
        return histograms_;
    }

    bool IsEmpty() const;
    float_type GetWeight() const;
    uint32_t Size() const { return row_indices_.size(); };

    uint32_t leaf_index_;
private:
    float_type CalculateGain(float_type gradient, float_type hessian) const;
    float_type CalculateWeight(float_type gradient, float_type hessian) const;
    float_type weight_;
    std::vector<uint32_t> row_indices_;
    Histogram* histograms_ = NULL;
    uint32_t histogram_count_;
    const std::vector<uint32_t>& bin_counts_;
    float_type lambda_l2_reg_;
};
#endif //CPP_LEAF_H

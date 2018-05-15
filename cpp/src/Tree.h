#ifndef CPP_TREE_H
#define CPP_TREE_H

#include <vector>
#include <memory>
#include <tuple>
#include <fstream>
#include "Config.h"
#include "Dataset.h"
#include "OptData.h"
#include "Leaf.h"

struct SearchParameters {
    SearchParameters() : gain(0), bin(0) {}

    float_type gain;
    bin_id bin;
    std::vector<float_type> left_weigths;
    std::vector<float_type> right_weights;
};

struct ThreadParameters{
    ThreadParameters(uint32_t thread_id,
                     uint32_t index_interval_start,
                     uint32_t index_interval_end) :
            thread_id(thread_id),
            index_interval_start(index_interval_start),
            index_interval_end(index_interval_end) {}
    uint32_t thread_id;
    uint32_t index_interval_start;
    uint32_t index_interval_end;
};

class Tree {
public:
    struct Split {
        Split() = default;
        Split(uint32_t feature, bin_id bin) : feature(feature), bin(bin) {}

        uint32_t feature;
        bin_id bin;
    };

    Tree(const Config& config) : initialized_(false), config_(config) {};

    void Construct(Dataset* dataset);

    void UpdatePredictions(Dataset* dataset) const;

    bool IsInitialized() const;

    void Save(std::ofstream& stream);
    void Load(std::ifstream& stream);

private:
    bool initialized_;
    const Config& config_;
    uint32_t depth_ = 0;
    std::vector<Split> splits_;
    std::vector<float_type> weights_;

    std::vector<uint32_t> SampleRows(uint32_t n_rows) const;
    void MakeNewLeafs(Dataset* dataset,
                            const std::vector<Leaf>& leafs,
                            uint32_t feature_number,
                            const SearchParameters& best_params,
                            std::vector<Leaf>* children,
                            uint32_t depth);

    void FindSplit(Dataset* dataset,
                   const std::vector<uint32_t>& indices,
                   ThreadParameters thread_params,
                   uint32_t depth,
                   const std::vector<Leaf>* parent_leafs,
                   std::vector<Leaf>* leafs,
                   std::vector<SearchParameters>* split_params);
};

#endif //CPP_TREE_H

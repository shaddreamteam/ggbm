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

class Tree {
public:
    struct Split {
        Split() = default;
        Split(uint32_t feature, bin_id bin) : feature(feature), bin(bin) {}

        uint32_t feature;
        bin_id bin;
    };

    Tree(const Config& config) : initialized_(false), config_(config) {};

    void Construct(const TrainDataset& dataset,
                   const std::vector<float_type>& gradients,
                   const std::vector<float_type>& hessians);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const;

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
    std::vector<Leaf> MakeNewLeafs(std::vector<bin_id> feature_vector,
                                   const std::vector<Leaf>& leafs,
                                   const SearchParameters& best_params);
};

#endif //CPP_TREE_H

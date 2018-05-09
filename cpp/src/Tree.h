#ifndef CPP_TREE_H
#define CPP_TREE_H

#include <vector>
#include <memory>
#include <tuple>
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
    Tree() : initialized_(false) {};

    void Construct(const Config& config,
                   const TrainDataset& dataset, 
                   const std::vector<float_type>& gradients,
                   const std::vector<float_type>& hessians);

    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const;
    std::vector<float_type> PredictFromFile(const std::string& filename, 
                                            const FeatureTransformer& ft, 
                                            bool fileHasTarget,
                                            char sep=',') const;
    bool IsInitialized() const;

private:
    bool initialized_;
    uint32_t depth_ = 0;
    std::vector<std::tuple<uint32_t, bin_id>> splits_;
    std::vector<float_type> weights_;

    std::vector<uint32_t> SampleRows(const Config& config,
                                     uint32_t n_rows) const;
    void CalculateSplitParameters(
        const Config& config,
        const TrainDataset& dataset,
        const std::vector<Leaf>& leafs, 
        const std::vector<float_type>& gradients,
        const std::vector<float_type>& hessians,
        uint32_t feature_number,
        std::vector<SearchParameters>* split_params);
};

#endif //CPP_TREE_H

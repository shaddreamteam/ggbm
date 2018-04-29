#ifndef CPP_TREE_H
#define CPP_TREE_H

#include <vector>
#include <memory>
#include <tuple>
#include "Dataset.h"
#include "GGBM.h"
#include "Leaf.h"

class Tree {
public:

	// constructor to test predict with given weights
    Tree(uint32_t max_depth, std::vector<float_type> weights, std::vector<std::tuple<uint32_t, bin_id>> splits)
      : initialized_(true), max_depth_(max_depth), depth_(max_depth),
        weights_(weights), splits_(splits) {};


    Tree(uint32_t max_depth) : initialized_(false), max_depth_(max_depth) {};
    void Construct(std::shared_ptr<const Dataset> dataset, std::shared_ptr<const OptData> optData, float_type lambda_l2_reg);
    std::vector<float_type> PredictFromBins(const std::vector<std::vector<bin_id>>& data_x_binned) const;
    std::vector<float_type> PredictFromFile(const std::string& filename, std::shared_ptr<const Dataset> dataset, char sep=',') const;

private:
    bool initialized_;
    uint32_t max_depth_;
    uint32_t depth_ = 0;
    std::vector<std::tuple<uint32_t, bin_id>> splits_;
    std::vector<float_type> weights_;

    std::tuple<float, std::vector<float_type>> GetSplitResult(uint32_t feature_number);
};

#endif //CPP_TREE_H

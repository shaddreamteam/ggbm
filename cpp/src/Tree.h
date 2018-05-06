#ifndef CPP_TREE_H
#define CPP_TREE_H

#include <vector>
#include <memory>
#include <tuple>
#include "Dataset.h"
#include "OptData.h"
#include "Leaf.h"

class Tree {
public:

	// constructor to test predict with given weights
    Tree(uint32_t max_depth,
         std::vector<float_type> weights,
         std::vector<std::tuple<uint32_t, bin_id>> splits,
         uint32_t thread_count) : initialized_(true),
                                  max_depth_(max_depth),
                                  depth_(max_depth),
                                  splits_(std::move(splits)),
                                  weights_(std::move(weights)),
                                  thread_count_(thread_count) {};


    Tree(uint32_t max_depth, uint32_t thread_count) : initialized_(false),
                                                      max_depth_(max_depth),
                                                      thread_count_(thread_count) {};
    void Construct(const TrainDataset& dataset, 
                   const std::vector<float_type>& gradients,
                   const std::vector<float_type>& hessians,
                   float_type lambda_l2_reg,
                   float_type row_sampling,
                   uint32_t min_subsample);
    std::vector<float_type> PredictFromDataset(const Dataset& dataset) const;
    std::vector<float_type> PredictFromFile(const std::string& filename, 
                                            const FeatureTransformer& ft, 
                                            bool fileHasTarget,
                                            char sep=',') const;
    bool IsInitialized() const;

private:
    bool initialized_;
    uint32_t max_depth_;
    uint32_t depth_ = 0;
    std::vector<std::tuple<uint32_t, bin_id>> splits_;
    std::vector<float_type> weights_;
    uint32_t thread_count_;

    std::tuple<float, std::vector<float_type>> GetSplitResult(uint32_t feature_number);
};

#endif //CPP_TREE_H

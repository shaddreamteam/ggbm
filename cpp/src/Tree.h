#include <vector>
#include <memory>
#include <tuple>
#include "Dataset.h"
#include "GGBM.h"
#include "Leaf.h"

class Tree {
public:
    Tree(uint32_t max_depth) : initialized_(false), max_depth_(max_depth) {};
    void Construct(std::shared_ptr<const Dataset> dataset, std::shared_ptr<const OptData> optData, float_type lambda_l2_reg);
    float_type Predict(std::vector<bin_id> featres) const;

private:
    bool initialized_;
    uint32_t max_depth_;
    uint32_t depth_;
    std::vector<std::tuple<uint32_t, bin_id>> splits_;
    std::vector<float_type> weights_;

    std::tuple<float, std::vector<float_type>> GetSplitResult(uint32_t feature_number);
};

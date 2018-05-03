#ifndef CPP_FEATURETRANSFORMER_H
#define CPP_FEATURETRANSFORMER_H

#include <vector>
#include "Base.h"

class FeatureTransformer {
public:
    //static std::tuple<std::vector<char>, std::vector<float_type>, std::vector<float_type>>
    FeatureTransformer(uint32_t thread_count) : initialized_(false),
                                                thread_count_(thread_count) {};

    std::vector<std::vector<bin_id>>
    FitTransform(const std::vector<std::vector<float_type>>& feature_values);

    std::vector<std::vector<bin_id>>
    Transform(const std::vector<std::vector<float_type>>& feature_values) const;
    
    uint32_t GetBinCount(uint32_t feature_number) const;
private:
/*    std::vector<float_type> GreedyFindBin(const std::vector<float_type>& distinct_values,
                                          const std::vector<uint32_t>& counts,
                                          uint64_t total_cnt);*/

    bool initialized_;
    uint32_t thread_count_;
    std::vector<std::vector<float_type>> bin_upper_bounds_;
};


#endif //CPP_FEATURETRANSFORMER_H

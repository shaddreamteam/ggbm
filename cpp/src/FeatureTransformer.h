#ifndef CPP_FEATURETRANSFORMER_H
#define CPP_FEATURETRANSFORMER_H

#include <vector>
#include <fstream>
#include "Base.h"
#include "Config.h"

class FeatureTransformer {
public:
    //static std::tuple<std::vector<char>, std::vector<float_type>, std::vector<float_type>>
    FeatureTransformer(const Config& config) : initialized_(false),
                                               config_(config) {};

    std::vector<std::vector<bin_id>>
    FitTransform(const std::vector<std::vector<float_type>>& feature_values);

    std::vector<std::vector<bin_id>>
    Transform(const std::vector<std::vector<float_type>>& feature_values) const;
    
    uint32_t GetBinCount(uint32_t feature_number) const;

    void Save(std::ofstream& stream);
    void Load(std::ifstream& stream);
private:
/*    std::vector<float_type> GreedyFindBin(const std::vector<float_type>& distinct_values,
                                          const std::vector<uint32_t>& counts,
                                          uint64_t total_cnt);*/

    bool initialized_;
    const Config& config_;
    std::vector<std::vector<float_type>> bin_upper_bounds_;
};


#endif //CPP_FEATURETRANSFORMER_H

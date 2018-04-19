#include "Dataset.h"

Dataset::Dataset(const std::string& filename, uint32_t thread_count) :
        ft_(FeatureTransformer(thread_count)) {
    targets_ = data_y;
    feature_bin_ids_ = ft_.FitTransform(data_x);
}

bin_id Dataset::GetFeature(uint32_t row_number, uint32_t feature_number) const {
    return feature_bin_ids_[feature_number][row_number];
}

float_type Dataset::GetTarget(uint32_t row_number) const {
    return targets_[row_number];
}

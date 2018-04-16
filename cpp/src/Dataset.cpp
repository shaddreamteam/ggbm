#include "Dataset.h"

Dataset::Dataset(const std::string& filename, uint32_t thread_count) :
        ft_(FeatureTransformer(thread_count)) {
    targets_ = data_y;
    feature_bin_ids_ = ft_.FitTransform(data_x);
}

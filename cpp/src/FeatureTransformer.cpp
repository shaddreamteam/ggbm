#include <exception>
#include <algorithm>
#include <cmath>
#include "FeatureTransformer.h"

std::vector<std::vector<bin_id>>
FeatureTransformer::FitTransform(const std::vector<std::vector<float_type>>& feature_values) {
    std::vector<std::vector<bin_id>> transform_result;
    for (auto feature_vector: feature_values) { // copy to sort
        std::sort(feature_vector.begin(), feature_vector.end());
        std::vector<float_type> distinct_values = {feature_vector[0]};
        std::vector<uint32_t> counts = {1};
        for (uint32_t i = 1; i < feature_vector.size(); ++i) {
            if (feature_vector[i] == distinct_values.back()) {
                ++counts.back();
            } else {
                distinct_values.push_back(feature_vector[i]);
                counts.push_back(1);
            }
        }
        uint64_t row_count = feature_vector.size();
        bin_upper_bounds_.push_back(GreedyFindBin(distinct_values, counts, row_count));
    }

    initialized_ = true;
    return Transform(feature_values);
}

std::vector<std::vector<bin_id>>
FeatureTransformer::Transform(const std::vector<std::vector<float_type>>& feature_values) const{
    if (!initialized_) {
        throw std::runtime_error("Not initialized");
    }
    std::vector<std::vector<bin_id>> transform_res;
    transform_res.reserve(feature_values.size());
    for (uint32_t index = 0; index < feature_values.size(); ++index) {
        std::vector<bin_id> feature_bins;
        feature_bins.reserve(feature_values[index].size());
        for (auto value: feature_values[index]) {
            bin_id bin = std::upper_bound(bin_upper_bounds_[index].begin(),
                                          bin_upper_bounds_[index].end(),
                                          value) - bin_upper_bounds_[index].begin();
            feature_bins.push_back(bin);
        }
        transform_res.push_back(feature_bins);
    }

    return transform_res;
}

std::vector<float_type>
FeatureTransformer::GreedyFindBin(const std::vector<float_type>& distinct_values,
                                  const std::vector<uint32_t>& counts,
                                  uint64_t total_cnt) {
    std::vector<float_type> bin_upper_bound;
    auto num_distinct_values = distinct_values.size();

    if (num_distinct_values <= kMaxBin) {
        int cur_cnt_inbin = 0;
        for (int i = 0; i < num_distinct_values - 1; ++i) {
            bin_upper_bound.push_back(std::nextafter(
                    (distinct_values[i] + distinct_values[i + 1]) / 2.0, INFINITY));
        }
        bin_upper_bound.push_back(std::numeric_limits<float_type>::infinity());
    } else {
        double mean_bin_size = static_cast<double>(total_cnt) / kMaxBin;

        // mean size for one bin
        uint64_t rest_bin_cnt = kMaxBin;
        uint64_t rest_sample_cnt = static_cast<int>(total_cnt);
        std::vector<bool> is_big_count_value(num_distinct_values, false);
        for (int i = 0; i < num_distinct_values; ++i) {
            if (counts[i] >= mean_bin_size) {
                is_big_count_value[i] = true;
                --rest_bin_cnt;
                rest_sample_cnt -= counts[i];
            }
        }

        mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
        std::vector<float_type> upper_bounds(kMaxBin, std::numeric_limits<float_type>::infinity());
        std::vector<float_type> lower_bounds(kMaxBin, std::numeric_limits<float_type>::infinity());

        uint32_t bin_cnt = 0;
        lower_bounds[bin_cnt] = distinct_values[0];
        uint32_t cur_cnt_inbin = 0;
        for (int i = 0; i < num_distinct_values - 1; ++i) {
            if (!is_big_count_value[i]) {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];
            // need a new bin
            if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
                (is_big_count_value[i + 1] &&
                        cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
                upper_bounds[bin_cnt] = distinct_values[i];
                ++bin_cnt;
                lower_bounds[bin_cnt] = distinct_values[i + 1];
                if (bin_cnt >= kMaxBin - 1) {
                    break;
                }
                cur_cnt_inbin = 0;
                if (!is_big_count_value[i]) {
                    --rest_bin_cnt;
                    mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
                }
            }
        }
        ++bin_cnt;
        // update bin upper bound
        bin_upper_bound.clear();
        for (int i = 0; i < bin_cnt - 1; ++i) {
            auto val = std::nextafter((upper_bounds[i] + lower_bounds[i + 1]) / 2.0, INFINITY);
            if (bin_upper_bound.empty() ||
                    val >= std::nextafter(bin_upper_bound.back(), INFINITY)) {
                bin_upper_bound.push_back(val);
            }
        }
        // last bin upper bound
        bin_upper_bound.push_back(std::numeric_limits<float_type>::infinity());
    }
    return bin_upper_bound;
}

uint32_t FeatureTransformer::GetBinCount(uint32_t feature_number) const {
    return bin_upper_bounds_.at(feature_number).size();
}

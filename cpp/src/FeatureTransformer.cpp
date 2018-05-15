#include <exception>
#include <algorithm>
#include <cmath>
#include <queue>
#include <thread>
#include <mutex>

#include "FeatureTransformer.h"
#include "TaskQueue.h"

std::vector<float_type>
GreedyFindBin(const std::vector<float_type>& distinct_values,
              const std::vector<uint32_t>& counts,
              uint64_t total_cnt);


std::vector<std::vector<bin_id>>
FeatureTransformer::FitTransform(const std::vector<std::vector<float_type>>& feature_values) {
    std::vector<std::vector<bin_id>> transform_result;
    bin_upper_bounds_.resize(feature_values.size());

    auto find_bin_vector = [&feature_values, this](int32_t j) {
        auto feature_vector = feature_values[j]; // copy to sort
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
        this->bin_upper_bounds_[j] = GreedyFindBin(distinct_values, counts, row_count);
    };

    TaskQueue<decltype(find_bin_vector), int32_t> task_queue(config_.GetThreads(),
                                                             &find_bin_vector);
    for (int32_t j = 0; j < feature_values.size(); ++j) {
        task_queue.Add(j);
    }
    task_queue.Run();

    initialized_ = true;
    return Transform(feature_values);
}

std::vector<std::vector<bin_id>>
FeatureTransformer::Transform(const std::vector<std::vector<float_type>>& feature_values) const{
    if (!initialized_) {
        throw std::runtime_error("Not initialized");
    }
    std::vector<std::vector<bin_id>> transform_res;
    transform_res.resize(feature_values[0].size(), std::vector<bin_id>(feature_values.size()));

    auto transform_vector = [&feature_values, &transform_res, this](int32_t j) {
        for (uint32_t row_index = 0; row_index < feature_values[j].size(); ++row_index) {
            bin_id bin = std::upper_bound(this->bin_upper_bounds_[j].begin(),
                                          this->bin_upper_bounds_[j].end(),
                                          feature_values[j][row_index]) -
                                              this->bin_upper_bounds_[j].begin();
            transform_res[row_index][j] = bin;
        }
    };

    TaskQueue<decltype(transform_vector), int32_t> task_queue(config_.GetThreads(),
                                                              &transform_vector);
    for (int32_t j = 0; j < feature_values.size(); ++j) {
        task_queue.Add(j);
    }
    task_queue.Run();

    return transform_res;
}

std::vector<float_type>
GreedyFindBin(const std::vector<float_type>& distinct_values,
                                  const std::vector<uint32_t>& counts,
                                  uint64_t total_cnt) {
    std::vector<float_type> bin_upper_bound;
    auto num_distinct_values = distinct_values.size();

    if (num_distinct_values <= kMaxBin) {
        for (int i = 0; i < num_distinct_values - 1; ++i) {
            bin_upper_bound.push_back(std::nextafter(
                    (distinct_values[i] + distinct_values[i + 1]) / 2.0, INFINITY));
        }
        bin_upper_bound.push_back(std::numeric_limits<float_type>::max());
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
        std::vector<float_type> upper_bounds(kMaxBin, std::numeric_limits<float_type>::max());
        std::vector<float_type> lower_bounds(kMaxBin, std::numeric_limits<float_type>::max());

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
        bin_upper_bound.push_back(std::numeric_limits<float_type>::max());
    }
    return bin_upper_bound;
}

std::vector<uint32_t> FeatureTransformer::GetBinCounts() const {
    std::vector<uint32_t> bin_counts;
    for (uint32_t feature_number = 0; feature_number < bin_upper_bounds_.size(); ++feature_number) {
        bin_counts.push_back(bin_upper_bounds_[feature_number].size());
    }
    return bin_counts;
}

void FeatureTransformer::Save(std::ofstream& stream) {
    stream << bin_upper_bounds_.size() << "\n";
    for (const auto& feature_upper_bounds: bin_upper_bounds_) {
        stream << feature_upper_bounds.size() << "\n";
        for (const auto upper_bound: feature_upper_bounds) {
            stream << upper_bound << " ";
        }
        stream << "\n";
    }
}

void FeatureTransformer::Load(std::ifstream& stream) {
    uint32_t features_count;
    stream >> features_count;
    bin_upper_bounds_.resize(features_count);
    for (int32_t feature_number = 0; feature_number < features_count; ++feature_number) {
        uint32_t bin_count;
        stream >> bin_count;
        bin_upper_bounds_[feature_number].resize(bin_count);
        for (int32_t bin_number = 0; bin_number < bin_count; ++bin_number) {
            float_type upper_bound;
            stream >> upper_bound;
            bin_upper_bounds_[feature_number][bin_number] = upper_bound;
        }
    }
    initialized_ = true;
}

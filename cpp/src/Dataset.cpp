#include <algorithm>
#include <sstream>
#include "Dataset.h"

Dataset::Dataset(const std::string& filename, uint32_t thread_count) :
        ft_(FeatureTransformer(thread_count)) {
    std::vector<std::vector<float_type>> data_x;
    GetSampleAndTargetFromFile(filename, ',', &data_x, &targets_);
    feature_bin_ids_ = ft_.FitTransform(data_x);
}

bin_id Dataset::GetFeature(uint32_t row_number, uint32_t feature_number) const {
    return feature_bin_ids_[feature_number][row_number];
}

float_type Dataset::GetTarget(uint32_t row_number) const {
    return targets_[row_number];
}

uint32_t Dataset::GetBinCount(uint32_t feature_number) const {
    return ft_.GetBinCount(feature_number);
}

uint32_t Dataset::GetNRows() const {
    return targets_.size();
}

uint32_t Dataset::GetNFeatures() const{
    return feature_bin_ids_.size();
}

std::vector<std::vector<bin_id>> Dataset::Transform(
    const std::vector<std::vector<float_type>>& feature_values) const{
    return ft_.Transform(feature_values);
}

void Dataset::GetSampleAndTargetFromFile(std::string filename, char sep,
                                         std::vector<std::vector<float_type>>* feature_values,
                                         std::vector<float_type>* targets) const{

    // read data from file and store it into feature_values and targets
    // sep -- ',' or '\t'
    // feature_values -- vector of feature columns, n_features x n_samples
    // targets -- vector of taget values, n_samples

    std::ifstream file(filename);  
    std::string line;

    std::getline(file, line);
    size_t n = std::count(line.begin(), line.end(), sep);
    if(targets == nullptr) {
        ++n;
    }

    feature_values->reserve(n);
    std::istringstream ss(line);
    std::string token;

    if(targets != nullptr) {
        std::getline(ss, token, sep);
        targets->push_back(std::stof(token));
    }

    while(std::getline(ss, token, sep)) {
        feature_values->push_back({std::stof(token)});
    }

    while(std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        if(targets != nullptr) {
            std::getline(ss, token, sep);
            targets->push_back(std::stof(token));
        }

        size_t idx = 0;
        while(std::getline(ss, token, sep)) {
            (*feature_values)[idx].push_back(std::stof(token));
            ++idx;
        }
    }
}

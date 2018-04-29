#include <algorithm>
#include <sstream>
#include "Dataset.h"

bin_id Dataset::GetFeature(uint32_t row_number, uint32_t feature_number) const {
    return feature_bin_ids_[feature_number][row_number];
}



uint32_t Dataset::GetNRows() const {
    return feature_bin_ids_.at(0).size();
}

uint32_t Dataset::GetNFeatures() const{
    return feature_bin_ids_.size();
}

//std::vector<std::vector<bin_id>> Dataset::Transform(
//    const std::vector<std::vector<float_type>>& feature_values) const{
//    return ft_.Transform(feature_values);
//}

void Dataset::GetDataFromFile(std::string filename, char sep,
                                         std::vector<std::vector<float_type>>& feature_values,
                                         std::vector<float_type>* targets,
                                         bool hasTarget) const{

    // read data from file and store it into feature_values and targets
    // sep -- ',' or '\t'
    // feature_values -- vector of feature columns, n_features x n_samples
    // targets -- vector of taget values, n_samples

    std::ifstream file(filename);  
    std::string line;

    while(std::getline(file, line)) {
        if(feature_values.empty()) {
            size_t n = std::count(line.begin(), line.end(), sep);
            if(!hasTarget) {
                ++n;
            }
            feature_values = std::vector<std::vector<float_type>>(n);
        }
        std::istringstream ss(line);
        std::string token;

        if(hasTarget) {
            std::getline(ss, token, sep);
            if(targets != nullptr) {
                targets->push_back(std::stof(token));
            }
        }

        size_t idx = 0;
        while(std::getline(ss, token, sep)) {
            feature_values[idx].push_back(std::stof(token));
            ++idx;
        }
    }
}

TrainDataset::TrainDataset(const std::string& filename, 
                      FeatureTransformer& ft) {
    std::vector<std::vector<float_type>> data_x;
    GetDataFromFile(filename, ',', data_x, &targets_, true);
    feature_bin_ids_ = ft.FitTransform(data_x);
    bin_counts_ = std::vector<uint32_t>(feature_bin_ids_.size(), 0);
    for(uint32_t i = 0; i < bin_counts_.size(); ++i) {
        bin_counts_[i] = ft.GetBinCount(i);
    }
}

float_type TrainDataset::GetTarget(uint32_t row_number) const {
    return targets_[row_number];
}

uint32_t TrainDataset::GetBinCount(uint32_t feature_number) const {
    return bin_counts_[feature_number];
}

TestDataset::TestDataset(const std::string& filename,
                         const FeatureTransformer& ft, bool fileHasTarget) {
    std::vector<std::vector<float_type>> data_x;
    GetDataFromFile(filename, ',', data_x, nullptr, fileHasTarget);
    feature_bin_ids_ = ft.Transform(data_x);
}


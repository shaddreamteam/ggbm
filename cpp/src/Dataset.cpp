#include "Dataset.h"

Dataset::Dataset(const std::string& filename, uint32_t thread_count) :
        ft_(FeatureTransformer(thread_count)) {
    targets_ = data_y;
    feature_bin_ids_ = ft_.FitTransform(data_x);
}



void Dataset::GetSampleAndTargetFromFile(std::string filename, char sep,
                                         std::vector<std::vector<float_type>>* feature_values,
                                         std::vector<float_type>* targets) {

    // read data from file and store it into feature_values and targets
    // sep -- ',' or '\t'
    // feature_values -- vector of feature columns, n_features x n_samples
    // targets -- vector of taget values, n_samples

    std::ifstream file(filename);  
    std::string line;

    std::getline(file, line);
    size_t n = std::count(line.begin(), line.end(), sep);
    feature_values->reserve(n);
    std::istringstream ss(line);
    std::string token;
    std::getline(ss, token, sep);
    targets->push_back(std::stof(token));

    while(std::getline(ss, token, sep)) {
        feature_values->push_back({std::stof(token)});
    }

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        std::getline(ss, token, sep);
        targets->push_back(std::stof(token));

        size_t idx = 0;
        while(std::getline(ss, token, sep)) {
            (*feature_values)[idx].push_back(std::stof(token));
            ++idx;
        }
    }
}

#include <algorithm>
#include <exception>
#include <sstream>
#include "Dataset.h"

void CSVReader::GetDataFromFile(std::string filename,
                                std::vector<std::vector<float_type>>* feature_values,
                                bool hasTarget,
                                std::vector<float_type>* targets,
                                char sep) const {

    // read data from file and store it into feature_values and targets
    // feature_values -- vector of feature columns, n_features x n_samples
    // targets -- optional, vector of taget values, n_samples
    // sep -- optional, ',' or '\t'

    std::ifstream file(filename);
    std::string line;

    if(sep=='\0') {
        auto file_format = filename.substr(filename.size() - 3, 3);
        if(file_format == "csv") {
            sep = ',';
        } else if(file_format == "tsv") {
            sep = '\t';
        } else {
            throw std::runtime_error("Separator couldn't be derived from file extension.");
        }
    }

    while(std::getline(file, line)) {
        if(feature_values->empty()) {
            size_t n = std::count(line.begin(), line.end(), sep);
            if(!hasTarget) {
                ++n;
            }
            feature_values->resize(n);
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
            feature_values->at(idx).push_back(std::stof(token));
            ++idx;
        }
    }
}
Dataset::Dataset(const std::vector<std::vector<bin_id>>& row_bin_ids,
                 const std::vector<uint32_t> bin_counts,
                 const std::vector<float_type>* targets) : bin_counts_(bin_counts) {
    row_count_ = row_bin_ids.size();
    feature_count_ = row_bin_ids[0].size();
    data_rows_ = (DataRow *) calloc(row_count_, sizeof(DataRow));
    for (int32_t row_index = 0; row_index < row_count_; ++row_index) {
        data_rows_[row_index].bin_ids = (bin_id *) calloc(feature_count_, sizeof(bin_id));
        for (int32_t feature_index = 0; feature_index < feature_count_; ++feature_index) {
            data_rows_[row_index].bin_ids[feature_index] = row_bin_ids[row_index][feature_index];
        }
        if (targets) {
            data_rows_[row_index].target = targets->at(row_index);
        }
    }
};

Dataset::~Dataset() {
    for (int32_t row_index = 0; row_index < row_count_; ++row_index) {
        free(data_rows_[row_index].bin_ids);
    }
    free(data_rows_);
}

void Dataset::SetBasePrediction(float_type base_prediction) {
    for (uint32_t i = 0; i < GetRowCount(); ++i) {
        data_rows_[i].prediction = base_prediction;
    }
}

DataRow* Dataset::GetData() {
    return data_rows_;
}

uint32_t Dataset::GetRowCount() const {
    return row_count_;
}

uint32_t Dataset::GetFeatureCount() const{
    return feature_count_;
}

const std::vector<uint32_t>& Dataset::GetBinCounts() const {
    return bin_counts_;
}



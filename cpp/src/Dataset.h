#ifndef CPP_DATASET_H
#define CPP_DATASET_H

#include <vector>
#include <string>
#include <istream>
#include <fstream>
#include <memory>
#include <array>
#include "Base.h"
#include "FeatureTransformer.h"

class CSVReader {
public:
    CSVReader() = default;

    void GetDataFromFile(std::string filename,
                         std::vector<std::vector<float_type>>* feature_values,
                         bool hasTarget,
                         std::vector<float_type>* targets=nullptr,
                         char sep='\0') const;

};

struct DataRow {
    uint32_t leaf_index;
    float_type gradient;
    float_type hessian;
    float_type prediction;
    float_type target;
    bin_id* bin_ids;
};


class Dataset {
public:
    Dataset(const std::vector<std::vector<bin_id>>& row_bin_ids,
            const std::vector<uint32_t> bin_counts,
            const std::vector<float_type>* targets);

    ~Dataset();

    void SetBasePrediction(float_type base_prediction);

    DataRow* GetData();

    uint32_t GetRowCount() const;
    uint32_t GetFeatureCount() const;
    const std::vector<uint32_t>& GetBinCounts() const;

private:
    DataRow* data_rows_;
    uint32_t row_count_;
    uint32_t feature_count_;
    const std::vector<uint32_t> bin_counts_;
};


#endif //CPP_DATASET_H

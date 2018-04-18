#ifndef CPP_DATASET_H
#define CPP_DATASET_H

#include <vector>
#include <string>
#include <istream>
#include "Base.h"
#include "dataset_stub.h"
#include "FeatureTransformer.h"

class Row {
public:

private:
    float_type target_;
    float_type gradient_;
    float_type hessian_;
};


class Dataset {
public:
    Dataset(const std::string& filename, uint32_t thread_count);
    GetFeature(uint32_t feature_number, uint32_t row_number);
    GetTarget(uint32_t row_number);

private:
    std::vector<float_type> targets_;
    std::vector<std::vector<bin_id>> feature_bin_ids_;
    FeatureTransformer ft_;
};

class CSVReader {
public:
    CSVReader(std::istream& csv);

private:

};


#endif //CPP_DATASET_H

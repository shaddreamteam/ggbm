#ifndef CPP_DATASET_H
#define CPP_DATASET_H

#include <vector>
#include <string>
#include <istream>
#include "Base.h"
#include "dataset_stub.h"
#include "FeatureTransformer.h"

#ifdef DEBUG
#include <cmath>
#endif //DEBUG
//class Row {
//public:
//
//private:
//    float_type target_;
//    float_type gradient_;
//    float_type hessian_;
//};


class Dataset {
public:
    Dataset(const std::string& filename, uint32_t thread_count);
#ifdef DEBUG
    Dataset(std::vector<std::vector<bin_id>> feature_bin_ids) :
        feature_bin_ids_(feature_bin_ids),
        ft_(0) {}
#endif //DEBUG
    bin_id GetFeature(uint32_t row_number, uint32_t feature_number) const;
    float_type GetTarget(uint32_t row_number) const;
    uint32_t GetBinCount(uint32_t feature_number) const;
    uint32_t GetNRows() const;
    uint32_t GetNFeatures() const;

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

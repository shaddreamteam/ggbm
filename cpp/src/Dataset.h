#ifndef CPP_DATASET_H
#define CPP_DATASET_H

#include <vector>
#include <string>
#include <istream>
#include <fstream>
#include "Base.h"
#include "dataset_stub.h"
#include "FeatureTransformer.h"

class Dataset {
public:
    bin_id GetFeature(uint32_t row_number, uint32_t feature_number) const;
    uint32_t GetNRows() const;
    uint32_t GetFeatureCount() const;

//    std::vector<std::vector<float_type>> Test(const std::string& filename) {
//        std::vector<std::vector<float_type>> res;
//        GetSampleAndTargetFromFile(filename, ',', &res);
//        return res;
//    };

protected:
    std::vector<std::vector<bin_id>> feature_bin_ids_;
 
    Dataset() {};
    void GetDataFromFile(std::string filename,
                         std::vector<std::vector<float_type>>& feature_values,
                         bool isTargetFirst,
                         std::vector<float_type>* targets=nullptr,
                         char sep='\0') const;
};

class TrainDataset : public Dataset {
public:
    TrainDataset(const std::string& filename, FeatureTransformer& f);
    float_type GetTarget(uint32_t row_number) const;
    uint32_t GetBinCount(uint32_t feature_number) const;

private:
    std::vector<float_type> targets_;
    std::vector<uint32_t> bin_counts_;
};

class TestDataset : public Dataset {
public:
    TestDataset(const std::string& filename, const FeatureTransformer& ft, bool fileHasTarget);
};

class CSVReader {
public:
    CSVReader(std::istream& csv);

private:

};

#endif //CPP_DATASET_H

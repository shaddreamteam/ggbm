#ifndef CPP_DATASET_H
#define CPP_DATASET_H

#include <vector>
#include <string>
#include <istream>
#include <fstream>
#include <memory>
#include "Base.h"
#include "dataset_stub.h"
#include "FeatureTransformer.h"

class Dataset {
public:
    bin_id GetFeature(uint32_t row_number, uint32_t feature_number) const;
    uint32_t GetRowCount() const;
    uint32_t GetFeatureCount() const;

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
    TrainDataset(const std::string& filename,
                 std::shared_ptr<FeatureTransformer> ft);

    float_type GetTarget(uint32_t row_number) const;
    uint32_t GetBinCount(uint32_t feature_number) const;
    const std::vector<bin_id>& GetFeatureVector(uint32_t feature_number) const;

private:
    std::vector<float_type> targets_;
    std::vector<uint32_t> bin_counts_;
};

class TestDataset : public Dataset {
public:
    TestDataset(const std::string& filename,
                const std::shared_ptr<FeatureTransformer> ft,
                bool file_has_target);
};


#endif //CPP_DATASET_H

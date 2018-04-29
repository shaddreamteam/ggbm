#ifndef OPT_DATA_H
#define OPT_DATA_H
#include <vector>
#include "Base.h"
#include "Dataset.h"
#include "Loss.h"

class OptData {
public:
    OptData(const TrainDataset& dataset,
            const std::vector<float_type>& predictionsi,
            const Loss& loss); 

    float_type GetGradient(uint32_t row_number) const;
    float_type GetHessian(uint32_t row_number) const;    
    float_type GetPrediction(uint32_t row_number) const;    

private:
    std::vector<float_type> gradients_;
    std::vector<float_type> hessians_;
    std::vector<float_type> predictions_;
};
#endif //OPT_DATA_H

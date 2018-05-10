#ifndef OPT_DATA_H
#define OPT_DATA_H
#include <vector>
#include "Base.h"
#include "Dataset.h"
#include "Loss.h"

class OptData {
public:
    OptData(const TrainDataset& dataset,
            const std::vector<float_type>& predictions,
            const Loss& loss); 

    const std::vector<float_type>& GetGradients() const;
    const std::vector<float_type>& GetHessians() const;    
    float_type GetPrediction(uint32_t row_number) const;    
    void Update(const TrainDataset& dataset,
                const std::vector<float_type>& increment_predictions,
                const Loss& loss);

private:
    std::vector<float_type> gradients_;
    std::vector<float_type> hessians_;
    std::vector<float_type> predictions_;
};
#endif //OPT_DATA_H

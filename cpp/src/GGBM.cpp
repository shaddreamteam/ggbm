#include "GGBM.h"


void GGBM::Train(const std::shared_ptr<const TrainDataset> trainDataset,
        const Loss& loss, uint32_t depth, uint32_t n_estimators, 
        float_type lambda_l2_reg, float_type learning_rate) {
    learning_rate_ = learning_rate;

    float_type first_prediction = loss.GetFirstPrediction(*trainDataset);
    std::vector<float_type> predictions(trainDataset->GetNRows(), first_prediction);
    std::shared_ptr<OptData> optDataset = std::make_shared<OptData>(
            *trainDataset, predictions, loss);

    for(uint32_t tree_number = 0; tree_number < n_estimators; ++tree_number) {
        Tree tree(depth);
        tree.Construct(trainDataset, optDataset, lambda_l2_reg);
        if(tree.IsInitialized()) {
            trees_.push_back(tree);
            auto tree_predictions = tree.PredictFromDataset(*trainDataset);
            for(uint32_t row_number = 0; row_number < trainDataset->GetNRows();
                    ++row_number) {
                float_type old_prediction = optDataset->GetPrediction(row_number);
                float_type new_prediction = old_prediction + 
                    learning_rate_ * tree_predictions[row_number];
            }
        } else {
            //here should be some message
            break;
        }
    }
}

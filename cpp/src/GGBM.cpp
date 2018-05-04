#include "GGBM.h"


void GGBM::Train(const std::shared_ptr<const TrainDataset> trainDataset,
        const Loss& loss, uint32_t depth, uint32_t n_estimators,
        float_type lambda_l2_reg, float_type learning_rate,
        float_type row_sampling, uint32_t min_subsample) {
    learning_rate_ = learning_rate;

    base_prediction_ = loss.GetFirstPrediction(*trainDataset);
    std::vector<float_type> predictions(trainDataset->GetRowCount(), base_prediction_);
    std::shared_ptr<OptData> optDataset = std::make_shared<OptData>(
            *trainDataset, predictions, loss);

    for(uint32_t tree_number = 0; tree_number < n_estimators; ++tree_number) {
        Tree tree(depth, thread_count_);
        tree.Construct(trainDataset, optDataset, lambda_l2_reg, 
                row_sampling, min_subsample);
        if(tree.IsInitialized()) {
            trees_.push_back(tree);
            auto tree_predictions = tree.PredictFromDataset(*trainDataset);
            optDataset->Update(*trainDataset, tree_predictions, learning_rate,
                    loss);
        }
    }
}

std::vector<float_type> GGBM::PredictFromDataset(const Dataset& dataset) const {
    std::vector<float_type> predictions(dataset.GetRowCount(), base_prediction_);
    for(const Tree& tree : trees_) {
        std::vector<float_type> treePredictions = tree.PredictFromDataset(dataset);
        for(uint32_t i = 0; i < predictions.size(); ++i) {
            predictions[i] += treePredictions[i] * learning_rate_;
        }
    }
    return predictions;
}

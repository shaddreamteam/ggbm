#include "GGBM.h"


void GGBM::Train(const Config& config, const TrainDataset& trainDataset,
                 const Loss& loss) {
    learning_rate_ = config.GetLearningRate();

    base_prediction_ = loss.GetFirstPrediction(trainDataset);
    std::vector<float_type> predictions(trainDataset.GetRowCount(),
                                        base_prediction_);
    OptData optDataset(trainDataset, predictions, loss);

    for(uint32_t tree_number = 0; tree_number < config.GetNEstimators();
            ++tree_number) {
        Tree tree;
        tree.Construct(config,
                       trainDataset,
                       optDataset.GetGradients(), 
                       optDataset.GetHessians());

        if(tree.IsInitialized()) {
            trees_.push_back(tree);
            auto tree_predictions = tree.PredictFromDataset(trainDataset);
            optDataset.Update(trainDataset, tree_predictions, learning_rate_,
                               loss);
            std::cout << "Tree #" << tree_number << 
                " constructed" << std::endl;
        }
    }
}

std::vector<float_type> GGBM::PredictFromDataset(const Dataset& dataset) const {
    std::vector<float_type> predictions(dataset.GetRowCount(),
                                        base_prediction_);
    for(const Tree& tree : trees_) {
        std::vector<float_type> treePredictions =
            tree.PredictFromDataset(dataset);
        for(uint32_t i = 0; i < predictions.size(); ++i) {
            predictions[i] += treePredictions[i] * learning_rate_;
        }
    }
    if(objective_ == kLogLoss) {
        for(float_type& p : predictions) {
            p = Sigmoid(p);
        }
    }
    return predictions;
}

void GGBM::Save(std::ofstream& stream) {
    stream << base_prediction_ << '\n';
    stream << static_cast<int32_t>(objective_) << '\n';
    stream << trees_.size() << '\n';
    for(Tree& tree : trees_) {
        tree.Save(stream);
    }
}

void GGBM::Load(std::ifstream& stream) {
    uint32_t tree_count;
    int32_t objective;
    stream >> base_prediction_ >> objective >> tree_count;
    objective_ = static_cast<ObjectiveType>(objective);
    trees_.clear();
    trees_.resize(tree_count);
    for (uint32_t tree_number = 0; tree_number < tree_count; ++tree_number) {
        trees_[tree_number].Load(stream);
    }
}

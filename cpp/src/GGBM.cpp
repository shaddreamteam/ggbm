#include "GGBM.h"


void GGBM::Train(const TrainDataset& trainDataset) {
    Loss* loss;
    if (config_.GetObjective() == kMse) {
        loss = new MSE();
    } else {
        loss = new LogLoss();
    }
    base_prediction_ = loss->GetFirstPrediction(trainDataset);
    std::vector<float_type> predictions(trainDataset.GetRowCount(),
                                        base_prediction_);
    OptData optDataset(trainDataset, predictions, *loss);

    for(uint32_t tree_number = 0; tree_number < config_.GetNEstimators();
            ++tree_number) {
        Tree tree(config_);
        tree.Construct(trainDataset, optDataset.GetGradients(), optDataset.GetHessians());

        if(tree.IsInitialized()) {
            trees_.push_back(tree);
            auto tree_predictions = tree.PredictFromDataset(trainDataset);
            optDataset.Update(trainDataset, tree_predictions, *loss);
            std::cout << "Tree #" << tree_number << 
                " constructed" << std::endl;
        }
    }

    delete loss;
}

std::vector<float_type> GGBM::PredictFromDataset(const Dataset& dataset) const {
    std::vector<float_type> predictions(dataset.GetRowCount(),
                                        base_prediction_);
    for(const Tree& tree : trees_) {
        std::vector<float_type> treePredictions =
            tree.PredictFromDataset(dataset);
        for(uint32_t i = 0; i < predictions.size(); ++i) {
            predictions[i] += treePredictions[i];
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
    feature_transformer_->Save(stream);
    stream << base_prediction_ << '\n';
    stream << static_cast<int32_t>(objective_) << '\n';
    stream << trees_.size() << '\n';
    for(Tree& tree : trees_) {
        tree.Save(stream);
    }
}

void GGBM::Load(std::ifstream& stream) {
    feature_transformer_ = std::make_shared<FeatureTransformer>(config_);
    feature_transformer_->Load(stream);
    uint32_t tree_count;
    int32_t objective;
    stream >> base_prediction_ >> objective >> tree_count;
    objective_ = static_cast<ObjectiveType>(objective);
    trees_.clear();
    trees_.reserve(tree_count);
    for (uint32_t tree_number = 0; tree_number < tree_count; ++tree_number) {
        trees_.emplace_back(config_);
        trees_[tree_number].Load(stream);
    }
}

std::shared_ptr<FeatureTransformer> GGBM::GetFeatureTransformer() {
    return feature_transformer_;
}

#include "GGBM.h"


void GGBM::Train(Dataset* train_dataset) {
    Loss* loss;
    auto data = train_dataset->GetData();
    if (config_.GetObjective() == kMse) {
        loss = new MSE();
    } else {
        loss = new LogLoss();
    }
    loss->SetFirstPrediction(train_dataset);
    loss->SetGradientsAndHessians(train_dataset);

    for(uint32_t tree_number = 0; tree_number < config_.GetNEstimators(); ++tree_number) {
        Tree tree(config_);
        tree.Construct(train_dataset);

        if(tree.IsInitialized()) {
            trees_.push_back(tree);
            tree.UpdatePredictions(train_dataset);
            // TODO update loss in parallel
            loss->SetGradientsAndHessians(train_dataset);

            //if (config_.GetVerbose()) {
                std::cout << "Tree #" << tree_number <<
                          " constructed" << std::endl;
            //}
        }
    }

    delete loss;
}

std::vector<float_type> GGBM::PredictFromDataset(Dataset* dataset) const {
    std::vector<float_type> predictions(dataset->GetRowCount(),
                                        base_prediction_);
    auto data = dataset->GetData();
    for(const Tree& tree : trees_) {
        tree.UpdatePredictions(dataset);
    }

    for (uint32_t row_number; row_number < dataset->GetRowCount(); ++row_number) {
        predictions[row_number] += data[row_number].prediction;
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

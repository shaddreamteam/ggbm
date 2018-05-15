#include <cmath>
#include <exception>
#include "Loss.h"
#include "TaskQueue.h"

void Loss::check_correct_input(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions) const {
    if (dataset.GetRowCount() != predictions.size()) {
        throw std::runtime_error("Target size doesn't match prediction size");
    }
    if (dataset.GetRowCount() == 0) {
        throw std::runtime_error("No targets specified");
    }
}

void MSE::UpdateGradientsAndHessians(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions,
        std::vector<float_type>* gradients,
        std::vector<float_type>* hessians) {

    check_correct_input(dataset, predictions);
    if (gradients->size() == 0) {
        gradients->resize(predictions.size());
        hessians->resize(predictions.size(), 2.0f / dataset.GetRowCount());
    }

    auto update_gradients = [&dataset, &predictions, gradients, hessians, this]
            (ThreadParameters thread_params) {
        auto size = dataset.GetRowCount();
        for (uint32_t current_index = thread_params.index_interval_start;
             current_index < thread_params.index_interval_end;
             ++current_index) {
            (*gradients)[current_index] = 2 * (predictions.at(current_index) -
                        dataset.GetTarget(current_index)) / size;
            }
    };

    TaskQueue<decltype(update_gradients), ThreadParameters>
            hist_queue(config_.GetThreads(), &update_gradients);
    uint32_t start = 0;
    for (uint32_t part_number = 0; part_number < config_.GetThreads(); ++part_number) {
        ThreadParameters thread_params(start,
                                       dataset.GetRowCount() * (part_number + 1) /
                                               config_.GetThreads());
        hist_queue.Add(thread_params);
        // end is not included
        start = thread_params.index_interval_end;
    }
    hist_queue.Run();
}

float_type MeanTarget(const TrainDataset& dataset) {
    float_type mean = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        mean += dataset.GetTarget(i) / dataset.GetRowCount();
    }
    return mean;
}

float_type MSE::GetFirstPrediction(const TrainDataset& dataset) const{
    return MeanTarget(dataset);
}

float_type MSE::GetLoss(const TrainDataset& dataset, 
                        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    float_type loss = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        loss += (predictions[i] - dataset.GetTarget(i)) *
            (predictions[i] - dataset.GetTarget(i));
    }
    loss /= dataset.GetRowCount();
    return loss;
}

float_type Sigmoid(float_type logit) {
    return 1 / (1 + std::exp(-logit));
}

void LogLoss::UpdateGradientsAndHessians(
        const TrainDataset& dataset,
        const std::vector<float_type>& predictions,
        std::vector<float_type>* gradients,
        std::vector<float_type>* hessians) {
    check_correct_input(dataset, predictions);

    if (gradients->size() == 0) {
        gradients->resize(predictions.size());
        hessians->resize(predictions.size(), 2.0f / dataset.GetRowCount());
    }

    auto update_gradients = [&dataset, &predictions, gradients, hessians, this]
            (ThreadParameters thread_params) {
        auto size = dataset.GetRowCount();
        for (uint32_t current_index = thread_params.index_interval_start;
             current_index < thread_params.index_interval_end;
             ++current_index) {
            (*gradients)[current_index] = -(dataset.GetTarget(current_index) -
                    Sigmoid(predictions[current_index])) / size;
            float_type probability = Sigmoid(predictions[current_index]);
            (*hessians)[current_index] = probability * (1 - probability) / size;

        }
    };

    TaskQueue<decltype(update_gradients), ThreadParameters>
            hist_queue(config_.GetThreads(), &update_gradients);
    uint32_t start = 0;
    for (uint32_t part_number = 0; part_number < config_.GetThreads(); ++part_number) {
        ThreadParameters thread_params(start,
                                       dataset.GetRowCount() * (part_number + 1) /
                                       config_.GetThreads());
        hist_queue.Add(thread_params);
        // end is not included
        start = thread_params.index_interval_end;
    }
    hist_queue.Run();
}

float_type LogLoss::GetFirstPrediction(const TrainDataset& dataset) const{
    return MeanTarget(dataset);
}

float_type LogLoss::GetLoss(const TrainDataset& dataset, 
                        const std::vector<float_type>& predictions) const {
    check_correct_input(dataset, predictions);
    float_type loss = 0;
    for(uint32_t i = 0; i < dataset.GetRowCount(); ++i) {
        float_type probability = Sigmoid(predictions[i]);
        float_type target = dataset.GetTarget(i);
        loss -= target * log(probability) + 
            (1 - target) * log(1 - probability);
    }
    loss /= dataset.GetRowCount();
    return loss;
}


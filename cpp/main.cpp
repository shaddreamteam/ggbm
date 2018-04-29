#include <iostream>
#include "src/Dataset.h"

#include "src/Tree.h"



int main() {

    // uint32_t depth=2;
    // std::cout << "depth=" << depth << std::endl;
    // std::vector<float_type> test_weights={10,20,30,40};

    // for (auto& s: test_weights) {
    //     std::cout << s << ' ';
    // }
    // std::cout << std::endl;
    
    // std::vector<std::tuple<uint32_t, bin_id>> test_splits={};
    // test_splits.push_back(std::make_tuple(0,2));
    // test_splits.push_back(std::make_tuple(1,2));
    // for (auto& s: test_splits) {
    //     std::cout << std::get<0>(s) << ' ' << std::get<1>(s) << std::endl;
    // }

    // Tree tree(depth, test_weights, test_splits);

    // std::vector<std::vector<bin_id>> x_test_bins={
    //     {1, 1, 3, 3},
    //     {1, 3, 1, 3},
    // };

    // auto preds = tree.PredictFromBins(x_test_bins);

    // std::cout << "predictions: " << std::endl;
    // for(size_t i=0; i < preds.size(); ++i) {
    //     std::cout << int(x_test_bins[0][i]) << ' '
    //               << int(x_test_bins[1][i]) << ' '
    //               << preds[i] << std::endl;
    // }
    std::shared_ptr<const Dataset> dataset_p = std::make_shared<const Dataset>("train.csv", 1);
    float_type mean = 0;
    for(const float_type& target : dataset_p->targets_) {
        mean += target / dataset_p->GetNRows();
    }

    std::vector<float_type> predictions(dataset_p->GetNRows(), mean);
    std::vector<float_type> hessians(dataset_p->GetNRows(), 2.0 / dataset_p->GetNRows());
    std::vector<float_type> gradients(dataset_p->GetNRows(), 0);
    for(uint32_t i = 0; i < dataset_p->GetNRows(); ++i) {
        gradients[i] = 2 * (predictions[i] - dataset_p->GetTarget(i)) / dataset_p->GetNRows();
    }

    std::shared_ptr<OptData> optData_p = std::make_shared<OptData>(gradients,
            hessians, predictions);
    
    Tree tree(2);
    tree.Construct(dataset_p, optData_p, 0);

    auto preds = tree.PredictFromFile("test.csv", dataset_p);

    std::cout << "preds:" << std::endl;
    for(auto& pred : preds) {
        std::cout << mean + pred << ' ';
    }
    std::cout << std::endl;



    std::cout << "Hello, World!" << std::endl;
    return 0;
}

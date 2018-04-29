#include <iostream>
#include "src/Dataset.h"
#include "src/Loss.h"
#include "src/Tree.h"
#include "src/GGBM.h"


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
    FeatureTransformer ft(1);
    std::shared_ptr<const TrainDataset> dataset = 
        std::make_shared<const TrainDataset>("train.csv", ft);
    MSE loss;

    float_type first_prediction = loss.GetFirstPrediction(*dataset);
    std::vector<float_type> predictions(dataset->GetNRows(), first_prediction);

    std::shared_ptr<OptData> optData_p = std::make_shared<OptData>(*dataset,
            predictions, loss);
    
//    Tree tree(2);
//    tree.Construct(dataset, optData_p, 0);

//    auto preds = tree.PredictFromFile("test.csv", ft, false);


    GGBM ggbm;
    ggbm.Train(dataset, loss, 2, 3, 0, 1);
    auto preds = ggbm.PredictFromDataset(*dataset);

    std::cout << "preds:" << std::endl;
    for(auto& pred : preds) {
        std::cout << pred << ' ';
    }
    std::cout << std::endl;

    ggbm = GGBM();
    ggbm.Train(dataset, loss, 1, 3, 0, 1);
    std::cout << "preds:" << std::endl;
    for(auto& pred : preds) {
        std::cout << pred << ' ';
    }
    std::cout << std::endl;

    ggbm = GGBM();
    ggbm.Train(dataset, loss, 2, 10, 0, 0.1);
    std::cout << "preds:" << std::endl;
    for(auto& pred : preds) {
        std::cout << pred << ' ';
    }
    std::cout << std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}

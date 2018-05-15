#include "src/UtilityFlow.h"
#include <chrono>

class ExecutionTimer {
public:
    ExecutionTimer() : start_(std::chrono::high_resolution_clock::now()) {}

    void Reset() {
        start_ = std::chrono::high_resolution_clock::now();

    }

    long long GetMilliseconds() {
        auto done = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(done-start_).count();
    }

    void PrintMilliseconds() {
        std::cout << GetMilliseconds() << std::endl;
    }


private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main(int argc, char **argv) {
    ExecutionTimer et;
    // ./cpp mode=train filename_model=model.bst filename_train=../../shitty_python_prototypes/train.csv filename_test=../../shitty_python_prototypes/test.csv threads=2 objective=mse learning_rate=1 depth=2 n_estimators=1 lambda=0 row_subsampling=1 min_subsample=1 file_has_target=false
    UtilityFlow uf;
    uf.Start(argc, argv);
    et.PrintMilliseconds();

    return 0;
}

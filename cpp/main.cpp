#include "src/UtilityFlow.h"

int main(int argc, char **argv) {
    // ./cpp mode=train filename_model=model.bst filename_train=../../shitty_python_prototypes/train.csv filename_test=../../shitty_python_prototypes/test.csv threads=2 objective=mse learning_rate=1 depth=2 n_estimators=1 lambda=0 row_subsampling=1 min_subsample=1 file_has_target=false
    UtilityFlow uf;
    uf.Start(argc, argv);

    return 0;
}

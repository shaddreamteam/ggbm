#include <iostream>
#include "src/Dataset.h"

int main() {
    Dataset ds = Dataset("test.csv", 4);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
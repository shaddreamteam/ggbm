cmake_minimum_required(VERSION 3.5)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp src/Leaf.cpp src/Leaf.h src/GGBM.cpp src/GGBM.h src/Tree.cpp src/Tree.h src/Dataset.cpp src/Dataset.h src/Base.h src/FeatureTransformer.cpp src/FeatureTransformer.h src/dataset_stub.h)
add_executable(cpp ${SOURCE_FILES})

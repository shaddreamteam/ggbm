cmake_minimum_required(VERSION 3.5)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
#set(CMAKE_CXX_FLAGS "-Wall -Werror")

#set(CMAKE_BUILD_TYPE Debug)

#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fno-omit-frame-pointer -fsanitize=address,undefined")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -g -fno-omit-frame-pointer -fsanitize=address,undefined")

find_package (Threads)

set(SOURCE_FILES main.cpp src/Leaf.cpp src/Leaf.h src/TaskQueue.cpp src/TaskQueue.h src/GGBM.cpp src/GGBM.h src/Tree.cpp src/Tree.h src/Dataset.cpp src/Dataset.h src/Base.h src/FeatureTransformer.cpp src/FeatureTransformer.h src/dataset_stub.h src/Loss.h src/Loss.cpp src/OptData.h src/OptData.cpp src/InputParser.h src/InputParser.cpp)
add_executable(cpp ${SOURCE_FILES})
target_link_libraries (cpp ${CMAKE_THREAD_LIBS_INIT})
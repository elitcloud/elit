cmake_minimum_required(VERSION 3.8)
project(cpp)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cpp
        tokenizer.hpp
        tokenizer.cpp
)

add_executable(cpp ${SOURCE_FILES})
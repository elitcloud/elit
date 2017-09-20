cmake_minimum_required(VERSION 3.8)
project(cpp)

set(CMAKE_CXX_STANDARD 11)
#find_package(Boost 1.58.0)

set(SOURCE_FILES
        tokenizer/english_tokenizer_test.cpp
        string_utils.hpp string_utils.cpp
        io_utils.cpp io_utils.hpp
        tokenizer/token_utils.cpp tokenizer/token_utils.hpp
        tokenizer/english_tokenizer.hpp tokenizer/english_tokenizer.cpp)

add_executable(english_tokenizer_test ${SOURCE_FILES})

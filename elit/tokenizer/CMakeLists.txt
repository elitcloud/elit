cmake_minimum_required(VERSION 3.8)
project(cpp)

set(CMAKE_CXX_STANDARD 11)
#find_package(Boost 1.58.0)

set(SOURCE_FILES
        main.cpp
        string_utils.hpp string_utils.cpp
        io_utils.cpp io_utils.hpp
        english_tokenizer.hpp english_tokenizer.cpp global_const.h)

add_executable(cpp ${SOURCE_FILES})

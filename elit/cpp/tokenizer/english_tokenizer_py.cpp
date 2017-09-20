#include "english_tokenizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(english_tokenizer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("tokenize", &tokenize, "A function which adds two numbers");
}

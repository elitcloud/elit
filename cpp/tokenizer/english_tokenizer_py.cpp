#include "english_tokenizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(english_tokenizer, m) {
    m.doc() = "English Tokenizer";

    m.def("init", &init, "Initializes the English tokenizer");
    m.def("tokenize", &tokenize, "Splits a string into linguistic tokens");
    m.def("segment", &segment, "Separates tokens into sentences");
}

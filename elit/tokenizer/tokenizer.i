%module tokenizer

%{
    #define SWIG_FILE_WITH_INIT
    #include "tokenizer.hpp"
%}

%include "std_vector.i"
%include "std_string.i"

// Instantiate templates used by example
namespace std {
   %template(StringVector) vector<string>;
}

%include "tokenizer.hpp"
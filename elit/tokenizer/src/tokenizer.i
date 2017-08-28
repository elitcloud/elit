%module tokenizer

%{
    #include "tokenizer.hpp"
%}

%include "std_vector.i"
%include "std_string.i"

// Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
}

%include "tokenizer.hpp"
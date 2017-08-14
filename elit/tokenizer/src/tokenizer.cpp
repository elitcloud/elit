#include "tokenizer.hpp"

char * tokenize(char * name) {
  char * buf;
  buf = (char *) malloc(strlen("tokenizer, ") + strlen(name) + 1);

  sprintf(buf, "tokenizer, %s", name);
  return buf;
 }

std::vector<std::string> vectorize(std::string s) {
    std::vector<std::string> v;
    for (int i=0; i<10; i++)
        v.push_back(s);
    return v;
};


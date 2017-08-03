#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char * tokenize(char * name) {
  char * buf;
  buf = (char *) malloc(strlen("tokenizer, ") + strlen(name) + 1);

  sprintf(buf, "tokenizer, %s", name);
  return buf;
 }
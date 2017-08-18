/**
 * Copyright 2017, Emory University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Jinho D. Choi
 */
#include <set>
#include <regex>
#include <vector>
#include <iterator>
#include "string_utils.hpp"

/**
 * Tokenizes the input string into linguistic tokens and saves them into a vector.
 * @param s the input string.
 * @return a vector of tokens from the input string; if there is no valid token, an empty vector is returned.
 */
std::vector<std::string> tokenize(std::string s);

/**
 * Tokenizes s[begin:end] into linguistic tokens and adds them to the vector.
 * @param s the input string.
 * @param begin the beginning index of the string to be tokenized (inclusive).
 * @param end the ending index of the string to be tokenized (exclusive).
 * @param v the vector where tokens to be added.
 * @return non-zero if any token is added; otherwise, 0.
 */
int tokenize(std::string s, size_t begin, size_t end, std::vector<std::string> &v);

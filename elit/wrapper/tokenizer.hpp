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
#include <vector>
#include <string>

// ======================================== Tokenization ========================================

/** Tokenizes the specific string and returns a vector of linguistically motivated tokens. */
std::vector<std::string> tokenize(std::string s);

/**
 * Peforms v.append(s[begin_index:end_index]);
 * Returns true if s[begin_index:end_index] is valid; otherwise, false.
 */
bool append(std::vector<std::string> &v, std::string s, int begin_index, int end_index);










// ======================================== Utilities ========================================

/** Returns a string where all beginning and ending white spaces are trimmed from the specific string. */
std::string trim(std::string s);



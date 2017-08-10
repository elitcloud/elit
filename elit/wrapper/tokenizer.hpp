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
#include <string>
#include <vector>
#include <algorithm>

const std::string PROTOCOLS[] = {"http://","https://","ftp://","sftp://","ssh://"};

// ======================================== Tokenization ========================================

/**
 * Returns a vector of linguistic tokens from the specific string.
 * If there is no valid token, an empty vector is returned.
 */
std::vector<std::string> tokenize(std::string s);

/**
 * Appends tokens within s[begin_index:end_index] to the specific vector.
 * Returns true if any token is added; otherwise, false.
 */
bool tokenize_aux(std::vector<std::string> &v, std::string s, size_t begin_index, size_t end_index);

/** Returns the index where a hyperlink begins. */
size_t find_hyperlink(std::string s, size_t begin_index, size_t end_index);









// ======================================== Utilities ========================================

/** Returns a string where all beginning and ending white spaces are trimmed from the specific string. */
std::string trim(std::string s);

/** Returns s[begin_index:end_index]. */
std::string substr(std::string s, size_t begin_index, size_t end_index);

/** Returns s[begin_index:end_index] in upper-case. */
std::string toupper(std::string s, size_t begin_index, size_t end_index);

/** Returns s[begin_index:end_index] in lower-case. */
std::string tolower(std::string s, size_t begin_index, size_t end_index);

/*
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

#include <string>
#include <set>
#include <map>

/**
 * Reads the input file consisting of one key per line.
 * @param filename the name of the input file.
 * @return a set of keys.
 */
std::set<std::wstring> read_word_set(std::string filename);

std::map<std::wstring,std::vector<size_t>> read_concat_word_map(std::string filename);
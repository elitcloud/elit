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
#include <regex>
#include <iterator>

// ======================================== Constants ========================================

/** Network protocols. */
const std::regex PROTOCOLS(
    R"(([A-Za-z]{3,4})(:\/\/))");

/**
 * :smile: :hug: :pencil:
 * <3 </3 <\3
 * ): (: $: *:
 * )-: (-: $-: *-:
 * )^: (^: $^: *^:
 * :( :) :P :p :O :3 :| :/ :\ :$ :* :@
 * :-( :-) :-P :-p :-O :-3 :-| :-/ :-\ :-$ :-* :-@
 * :^( :^) :^P :^p :^O :^3 :^| :^/ :^\ :^$ :^* :^@
 */
const std::regex EMOTICONS(
    R"((\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(\s|[\!\.\?\,\;]|$))");

/**
 * jinho@elit.com
 * jinho.choi@elit.com
 * choi@elit.emory.edu
 * jinho:choi@0.0.0.0
 */
const std::regex EMAILS(
    R"(([A-Za-z0-9\-\._]+(:\S+)?@)((([A-Za-z0-9\-]+\.)+([A-Za-z]{2,3}))|(\d{1,3}(\.\d{1,3}){3})))");

/** &arrow; &#123; */
const std::regex HTML_ENTITIES(
    R"(\&([A-Za-z]+|(\#\d+))\;)");

/** #happy2017,@JinhoChoi */
const std::regex HASHTAGS(
    R"([\#\@]([A-Za-z][A-Za-z0-9_]+))");

// ======================================== Tokenization ========================================

/**
 * @param s string to be tokenized.
 * @return vector of ordered tokens from the string. If there is no valid token, an empty vector is returned.
 */
std::vector<std::string> tokenize(std::string s);

/**
 * @param v vector where tokens are added.
 * @param s string to be tokenized.
 * @param begin_index beginning index of the string to be processed (inclusive).
 * @param end_index ending index of the string to be processed (exclusive).
 * @return non-zero if any token is added; otherwise, 0.
 */
int tokenize_aux(std::vector<std::string> &v, std::string s, size_t begin_index, size_t end_index);

/**
 * Tokenizes using regular expressions: hyperlink, emoticon, email.
 * @param v vector where tokens are added.
 * @param s string to be tokenized.
 * @param begin_index beginning index of the string to be processed (inclusive).
 * @param end_index ending index of the string to be processed (exclusive).
 * @return non-zero if any hyperlink is added; otherwise, 0.
 */
int tokenize_regex(std::vector<std::string> &v, std::string s, size_t begin_index, size_t end_index);








// ======================================== Utilities ========================================

/** Returns a string where all beginning and ending white spaces are trimmed from the specific string. */
std::string trim(std::string s);

/** Returns s[begin_index:end_index]. */
std::string substr(std::string s, size_t begin_index, size_t end_index);

/** Returns s[begin_index:end_index] in upper-case. */
std::string toupper(std::string s, size_t begin_index, size_t end_index);

/** Returns s[begin_index:end_index] in lower-case. */
std::string tolower(std::string s, size_t begin_index, size_t end_index);

/**
 * Returns the beginning index of the source string that matches the target string.
 * If there is no match, return string::npos.
 */
size_t find(std::string source, std::string target, size_t source_begin, size_t source_end);

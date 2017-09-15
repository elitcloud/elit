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
#include "token_utils.hpp"
#include <regex>

// ======================================== Tokenization ========================================

/**
 * Tokenizes the input string into linguistic tokens and saves them into a vector.
 * @param s the input string (supports unicode characters).
 * @return the vector of pairs, where each pair consists of (token, beginning index).
 */
TokenList tokenize(std::wstring s);

/**
 * Tokenizes s[begin:end] into tokens and adds them to the output vector.
 * The actual tokenization happens in this function.
 * @param v the output vector.
 * @param s the input string.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true if any token in s[begin:end] is added to the output vector.
 */
bool tokenize_aux(TokenList &v, std::wstring s, size_t begin, size_t end);

/**
 * Tokenizes trivial cases where s[begin:end] should be considered as one token:
 * 1. Single character token (end - begin == 1).
 * 2. Token contains only alphabets or digits.
 * @param v the output vector.
 * @param s the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true if any token in s[begin:end] is added to the output vector.
 */
bool tokenize_trivial(TokenList &v, std::wstring s, size_t begin, size_t end);

// ======================================== Add Token ========================================

/**
 * Adds the input token to the output vector.
 * @param v the output vector.
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 */
void add_token(TokenList &v, std::wstring token, size_t begin, size_t end);

/** Private function: v.add(s[begin:end]). */

/**
 * Adds s[begin:end] to the output vector.
 * @param v the output vector.
 * @param s the input string.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 */
void add_token_aux(TokenList &v, std::wstring s, size_t begin, size_t end);

// ======================================== Concatenate ========================================

/**
 * Concatenates the input token to the previous token in the output vector.
 * @param v the output vector.
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true is the input token is concatenated to the previous token.
 */
bool concat_token(TokenList &v, std::wstring token, size_t begin, size_t end);

/**
 * @param prev w_{i-1}.
 * @param curr w_{i}.
 * @return true is the previous token is an apostrophe, and the current token is {"cause", "tis", ...}.
 */
bool is_concat_apostrophe_front(std::wstring prev, std::wstring curr);

/**
 * @param prev w_{i-1}.
 * @param curr w_{i}.
 * @return true is the previous token is an abbreviatioin, and the current token is a period.
 */
bool is_concat_abbreviation(std::wstring prev, std::wstring curr);

/**
 * PRE: curr.size() == 1.
 * e.g., "AB&CD", "a|b", "1/2", "a-b".
 * @param prev w_{i-1}.
 * @param curr w_{i}.
 * @param next w_{i+1}.
 * @return true if the current token is [&|/], and the previous and next tokens are either single characters or all uppercase alphabets.
 */
bool is_concat_acronym(std::wstring prev, std::wstring curr, std::wstring next);

/**
 * PRE: curr.size() == 1.
 * e.g., "mis-predict", "book-able".
 * @param prev w_{i-1}.
 * @param curr w_{i}.
 * @param next w_{i+1}.
 * @return true if the current token is a hyphen, and the previous or next token is a qualified prefix/suffix.
 */
bool is_concat_hyphen(std::wstring prev, std::wstring curr, std::wstring next);

/**
 * Concatenates "No" and "." preceeding a digit. (e.g., No. 1).
 * @param v the output vector
 * @param prev w_{i-1}.
 * @param curr w_{i}.
 * @param next w_{i+1}.
 * @return true if the previous token is "no", the current token is "." and the next token is "[:digit:]".
 */
bool concat_token_no(TokenList &v, std::wstring prev, std::wstring curr, std::wstring next);

// ======================================== Split ========================================

/**
 * Splits the input token if necessary and adds the sub-tokens.
 * @param v the output vector.
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true if sub-tokens are added to the output vector.
 */
bool split_token(TokenList &v, std::wstring token, size_t begin, size_t end);

/**
 * Splits the unit from the input token (e.g., 10cm).
 * @param v the output vector.
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true if sub-tokens are added to the output vector.
 */
bool split_token_unit(TokenList &v, std::wstring token, size_t begin, size_t end);

/**
 * Splits concatenated words from the input token (e.g., whadya -> wha d ya).
 * @param v the output vector.
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return true if sub-tokens are added to the output vector.
 */
bool split_token_concat_words(TokenList &v, std::wstring token, size_t begin, size_t end);

bool split_token_final_mark(TokenList &v, std::wstring token, size_t begin, size_t end);

// ======================================== Regular Expression ========================================

typedef void (* regex_aux)(TokenList &v, std::wstring s, size_t begin, size_t end, std::wsmatch m, size_t flag);

bool tokenize_regex(TokenList &v, std::wstring s, size_t begin, size_t end);

bool tokenize_regex_aux(TokenList &v, std::wstring s, size_t begin, size_t end, std::wregex r, regex_aux f, size_t flag=0);

void regex_group(TokenList &v, std::wstring s, size_t begin, size_t end, std::wsmatch m, size_t flag);

void regex_hyperlink(TokenList &v, std::wstring s, size_t begin, size_t end, std::wsmatch m, size_t flag=0);

// ======================================== Symbol ========================================

typedef bool (* symbol_aux_0)(wchar_t c);

typedef bool (* symbol_aux_1)(std::wstring s, size_t begin, size_t end, size_t curr, size_t last);

bool skip_symbol(std::wstring s, size_t begin, size_t end, size_t curr);

bool tokenize_symbol(TokenList &v, std::wstring s, size_t begin, size_t end);

bool tokenize_symbol(TokenList &v, std::wstring s, size_t begin, size_t end, size_t curr, symbol_aux_0 f0, symbol_aux_1 f1);

bool tokenize_symbol_true(std::wstring s, size_t begin, size_t end, size_t curr, size_t last);

bool tokenize_symbol_edge(std::wstring s, size_t begin, size_t end, size_t curr, size_t last);

bool tokenize_symbol_currency_like(std::wstring s, size_t begin, size_t end, size_t curr, size_t last);

bool is_separator(wchar_t c);

bool is_symbol_edge(wchar_t c);

bool is_currency_like(wchar_t c);

size_t get_last_sequence_index(std::wstring s, size_t curr, size_t end);

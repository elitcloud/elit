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

#include <tuple>
#include <string>
#include <vector>

// ======================================== Types ========================================

typedef std::tuple<std::wstring, size_t, size_t> Token;
typedef std::vector<Token> TokenList;

// ======================================== Token ========================================

/**
 * @param token the input token.
 * @param begin the begin index of the input token (inclusive).
 * @param end the end index of the input token (exclusive).
 * @return a new token consisting of all the field.
 */
Token create_token(std::wstring token, size_t begin, size_t end);

/**
 * @param t the input token.
 * @return the word-form of the input token.
 */
std::wstring get_form(Token t);

/**
 * @param t the input token.
 * @return the begin index of the input token.
 */
size_t get_begin(Token t);

/**
 * @param t the input token.
 * @return the end index of the input token.
 */
size_t get_end(Token t);

/**
 * @param t1 the first token.
 * @param t2 the second token.
 * @return true if two tokens have the exact same values for all fields; otherwise, false.
 */
bool equals(Token t1, Token t2);

/**
 * @param t1 the first token list.
 * @param t2 the second token list.
 * @return true if two token lists have the exact same tokens; otherwise, false.
 */
bool equals(TokenList t1, TokenList t2);
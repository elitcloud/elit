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

// ======================================== String ========================================

/**
 * @param s the input string.
 * @param begin the beginning index of the substring (inclusive).
 * @param end the ending index of the substring (exclusive).
 * @return s.substr(begin, end - begin).
 */
std::wstring substr(std::wstring s, size_t begin, size_t end);

/**
 * @param s the input string.
 * @return the string where all beginning and ending spaces are trimmed from the input string.
 */
std::wstring trim(std::wstring s);

/**
 * @param source the source string.
 * @param target the target string.
 * @param source_begin the beginning index of the source string to compare (inclusive).
 * @param source_end the ending index of the source string to compare (exlusive).
 * @return the beginning index of the source string that matches the target string; if no match, string::npos is returned.
 */
size_t find(std::string source, std::string target, size_t source_begin, size_t source_end);

// ======================================== Character ========================================

bool is_range(wchar_t c, wchar_t begin, wchar_t end);

bool is_single_quote(wchar_t c);

bool is_double_quote(wchar_t c);

bool is_left_bracket(wchar_t c);

bool is_right_bracket(wchar_t c);

bool is_bracket(wchar_t c);

bool is_arrow(wchar_t c);

bool is_hyphen(wchar_t c);

bool is_currency(wchar_t c);

bool is_final_mark(wchar_t c);

// ======================================== Encdoing ========================================

std::wstring utf8_to_wstring(const std::string& str);

std::string wstring_to_utf8(const std::wstring& str);
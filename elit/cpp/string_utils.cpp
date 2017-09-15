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
#include <boost/locale/encoding_utf.hpp>
#include "string_utils.hpp"

using namespace std;
using boost::locale::conv::utf_to_utf;

// ======================================== String ========================================

/**
 * @param s the input string.
 * @param begin the beginning index of the substring (inclusive).
 * @param end the ending index of the substring (exclusive).
 * @return s.substr(begin, end - begin).
 */
wstring substr(wstring s, size_t begin, size_t end)
{
    return s.substr(begin, end - begin);
}

/**
 * @param s the input string.
 * @return the string where all beginning and ending spaces are trimmed from the input string.
 */
wstring trim(wstring s)
{
    size_t idx;

    // trim beginning white spaces
    for (idx=0; idx<s.size(); idx++)
    {
        if (isspace(s[idx]) == 0)
        {
            if (idx > 0) s.erase(0, idx);
            break;
        }
    }

    // contain only white spaces
    if (idx == s.size())
        return L"";

    size_t lst = s.size() - 1;

    // trim ending white spaces
    for (idx=lst; idx>0; idx--)
    {
        if (isspace(s[idx]) == 0)
        {
            if (idx < lst) s.erase(idx+1, lst-idx);
            break;
        }
    }

    return s;
}

 /**
  * @param source the source string.
  * @param target the target string.
  * @param source_begin the beginning index of the source string to compare (inclusive).
  * @param source_end the ending index of the source string to compare (exlusive).
  * @return the beginning index of the source string that matches the target string; if no match, string::npos is returned.
  */
size_t find(string source, string target, size_t source_begin, size_t source_end)
{
    for (size_t i=source_begin; i+target.size()<=source_end; i++)
    {
        auto found = true;

        for (size_t j=0; j<target.size(); j++)
        {
            if (source[i+j] != target[j])
            {
                found = false;
                break;
            }
        }

        if (found) return i;
    }

    return string::npos;
}

// ======================================== Character ========================================

bool is_range(wchar_t c, wchar_t begin, wchar_t end)
{
    return begin <= c && c <= end;
}

bool is_single_quote(wchar_t c)
{
    return c == '\'' || c == '`' || is_range(c, L'\u2018', L'\u201B');
}

bool is_double_quote(wchar_t c)
{
    return c == '"' || is_range(c, L'\u201C', L'\u201F');
}

bool is_bracket(wchar_t c)
{
    return is_left_bracket(c) || is_right_bracket(c);
}

bool is_left_bracket(wchar_t c)
{
    return c == '(' || c == '{' ||c == '[' ||c == '<';
}

bool is_right_bracket(wchar_t c)
{
    return c == ')' || c == '}' ||c == ']' ||c == '>';
}

bool is_arrow(wchar_t c)
{
    return is_range(c, L'\u2190', L'\u21FF') || is_range(c, L'\u27F0', L'\u27FF') || is_range(c, L'\u2900', L'\u297F');
}

bool is_hyphen(wchar_t c)
{
    return c == '-' || is_range(c, L'\u2010', L'\u2014');
}

bool is_currency(wchar_t c)
{
    return c == '$' || is_range(c, L'\u00A2', L'\u00A5') || is_range(c, L'\u20A0', L'\u20CF');
}

bool is_final_mark(wchar_t c)
{
    return c == '.' || c == '?' || c == '!' || c == L'\u203C' || is_range(c, L'\u2047', L'\u2049');
}

// ======================================== Encoding ========================================

wstring utf8_to_wstring(const string& str)
{
    return utf_to_utf<wchar_t>(str.c_str(), str.c_str() + str.size());
}

string wstring_to_utf8(const wstring& str)
{
    return utf_to_utf<char>(str.c_str(), str.c_str() + str.size());
}
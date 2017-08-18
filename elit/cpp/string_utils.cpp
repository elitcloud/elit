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
#include "string_utils.hpp"
using namespace std;

/**
 * @param s the input string.
 * @return the string where all beginning and ending spaces are trimmed from the input string.
 */
string trim(string s)
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
        return "";

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
 * @param s the input string.
 * @param begin the beginning index of the substring (inclusive).
 * @param end the ending index of the substring (exclusive).
 * @return s.substr(begin, end - begin).
 */
string substr(string s, size_t begin, size_t end)
{
    return s.substr(begin, end - begin);
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

///**
// * @param s the input string.
// * @param begin the beginning index of the string to convert (inclusive).
// * @param end the ending index of the string to convert (exclusive).
// * @return
// */
//string toupper(string s, size_t begin, size_t end)
//{
//    string t(end - begin, '\0');
//
//    for (size_t i=begin; i<end; i++)
//        t[i-begin] = toupper(s[i]);
//
//    return t;
//}
//
///** Returns s[begin:end] in lower-case. */
//string tolower(string s, size_t begin, size_t end)
//{
//    string t(end - begin, '\0');
//
//    for (size_t i=begin; i<end; i++)
//        t[i-begin] = tolower(s[i]);
//
//    return t;
//}
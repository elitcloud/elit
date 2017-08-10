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
#include "tokenizer.hpp"
using namespace std;

// ======================================== Tokenization ========================================

/**
 * Returns a vector of linguistic tokens from the specific string.
 * If there is no valid token, an empty vector is returned.
 */
vector<string> tokenize(string s)
{
    size_t begin_index = 0, end_index = 1;
    vector<string> v;
    
    s = trim(s);
    
    for (; end_index<s.size(); end_index++)
    {
        if (isspace(s[end_index]))
        {
            tokenize_aux(v, s, begin_index, end_index);
            begin_index = end_index + 1;
        }
    }
    
    tokenize_aux(v, s, begin_index, end_index);
    return v;
}

/**
 * Appends tokens within s[begin_index:end_index] to the specific vector.
 * Returns true if any token is added; otherwise, false.
 */
bool tokenize_aux(vector<string> &v, string s, size_t begin_index, size_t end_index)
{
    // out of range
    if (end_index > s.size() || begin_index >= end_index)
        return false;
    
    // tokenize hyperlink
    size_t idx = find_hyperlink(s, begin_index, end_index);
    
    if (idx == 0)   // the entire substring is a hyperlink
    {
        v.push_back(substr(s, begin_index, end_index));
        return true;
    }
    else if (idx != string::npos)
    {
        tokenize_aux(v, s, begin_index, idx);
        v.push_back(substr(s, idx, end_index));
        return true;
    }
    

//    string lower = tolower(s, begin_index, end_index);
//    
//    if (PRESERVE.find(lower) != PRESERVE.end())
//    {
//        v.push_back(substr(s, begin_index, end_index));
//        return true;
//    }
    
    
    
    
    
    
    
    v.push_back(substr(s, begin_index, end_index));
    return true;
}

/** Returns the index where a hyperlink begins. */
size_t find_hyperlink(string s, int begin_index, int end_index)
{
    size_t idx;
    
    for (string p : PROTOCOLS)
    {
        idx = s.find(p, begin_index);
        
        if (idx != string::npos)
            return idx;
    }
    
    return string::npos;
}

// ======================================== Utilities ========================================

/** Returns a string where all beginning and ending white spaces are trimmed from the specific string. */
string trim(string s)
{
    size_t idx;
    
    // trim beginning white spaces
    for (idx=0; idx<s.size(); idx++)
    {
        if (!isspace(s[idx]))
        {
            if (idx > 0) s.erase(0, idx);
            break;
        }
    }
    
    // contain only white spaces
    if (idx == s.size())
        return "";
    
    size_t lst = s.size() - 1;
    
    // trim endding white spaces
    for (idx=lst; idx>0; idx--)
    {
        if (!isspace(s[idx]))
        {
            if (idx < lst) s.erase(idx+1, lst-idx);
            break;
        }
    }
    
    return s;
}

/** Returns s[begin_index:end_index]. */
string substr(string s, size_t begin_index, size_t end_index)
{
    return s.substr(begin_index, end_index - begin_index);
}

/** Returns s[begin_index:end_index] in upper-case. */
string toupper(string s, size_t begin_index, size_t end_index)
{
    string t(end_index - begin_index, '\0');
    
    for (size_t i=begin_index; i<end_index; i++)
        t[i-begin_index] = toupper(s[i]);
    
    return t;
}

/** Returns s[begin_index:end_index] in lower-case. */
string tolower(string s, size_t begin_index, size_t end_index)
{
    string t(end_index - begin_index, '\0');
    
    for (size_t i=begin_index; i<end_index; i++)
        t[i-begin_index] = tolower(s[i]);
    
    return t;
}

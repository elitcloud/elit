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
#include <iostream>
#include "tokenizer.hpp"
using namespace std;

// ======================================== Constants ========================================

/** Network protocols. */
const regex PROTOCOLS(
    "(([A-Za-z]{3,4})(:\\/\\/))");

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
const regex EMOTICONS(
    "((\\:\\w+\\:|\\<[\\/\\]?3|[\\(\\)\\\\D|\\*\\$][\\-\\^]?[\\:\\;\\=]|[\\:\\;\\=B8][\\-\\^]?[3DOPp\\@\\$\\*\\\\)\\(\\/\\|])(\\s|[\\!\\.\\?\\,\\;]|$))");

/**
 * jinho@elit.com
 * jinho.choi@elit.com
 * choi@elit.emory.edu
 * jinho:choi@0.0.0.0
 */
const regex EMAILS(
    "(([A-Za-z0-9\\-\\._]+(:\\S+)?@)((([A-Za-z0-9\\-]+\\.)+([A-Za-z]{2,3}))|(\\d{1,3}(\\.\\d{1,3}){3})))");

/** &arrow; &#123; */
const regex HTML_ENTITIES(
    "(\\&([A-Za-z]+|(\\#\\d+))\\;)");

/** #happy2017,@JinhoChoi */
const regex HASHTAGS(
    "([\\#\\@]([A-Za-z][A-Za-z0-9_]+))");


// ======================================== Tokenization ========================================

/**
 * @param s string to be tokenized.
 * @return vector of ordered tokens from the string. If there is no valid token, an empty vector is returned.
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
 *
 * @param v vector where tokens are added.
 * @param s string to be tokenized.
 * @param begin_index beginning index of the string to be processed (inclusive).
 * @param end_index ending index of the string to be processed (exclusive).
 * @return non-zero if any token is added; otherwise, 0.
 */
int tokenize_aux(vector<string> &v, string s, size_t begin_index, size_t end_index)
{
    // out of range
    if (end_index > s.size() || begin_index >= end_index)
        return 0;

    if (tokenize_regex(v, s, begin_index, end_index))
        return 1;


//    string lower = tolower(s, begin_index, end_index);
//    
//    if (PRESERVE.find(lower) != PRESERVE.end())
//    {
//        v.push_back(substr(s, begin_index, end_index));
//        return true;
//    }



    
    
    v.push_back(substr(s, begin_index, end_index));
    return 10;
}

/** Auxiliary function to tokenize_regex(). */
bool tokenize_regex_aux(vector<string> &v, string s, size_t begin_index, size_t end_index, string sub, regex r)
{
    smatch m;

    if (regex_search(sub, m, r))
    {
        tokenize_aux(v, s, begin_index, begin_index+m.position(0));
        v.push_back(m[0].str());
        tokenize_aux(v, s, begin_index+m.position(0)+m[0].str().size(), end_index);
        return true;
    }

    return false;
}

/**
 * Tokenizes using regular expressions: hyperlink, emoticon, email.
 * @param v vector where tokens are added.
 * @param s string to be tokenized.
 * @param begin_index beginning index of the string to be processed (inclusive).
 * @param end_index ending index of the string to be processed (exclusive).
 * @return non-zero if any hyperlink is added; otherwise, 0.
 */
int tokenize_regex(vector<string> &v, string s, size_t begin_index, size_t end_index)
{
    string sub = substr(s, begin_index, end_index);
    smatch m;

    // html entity
    if (tokenize_regex_aux(v, s, begin_index, end_index, sub, HTML_ENTITIES))
        return 1;

    // network protocol
    if (regex_search(sub, m, PROTOCOLS))
    {
        if (m.position(0) > 0)
        {
            auto idx = begin_index + m.position(0);
            tokenize_aux(v, s, begin_index, idx);
            v.push_back(substr(s, idx, end_index));
        }
        else
            v.push_back(sub);

        return 2;
    }

    // email
    if (tokenize_regex_aux(v, s, begin_index, end_index, sub, EMAILS))
        return 3;

    // hashtag
    if (tokenize_regex_aux(v, s, begin_index, end_index, sub, HASHTAGS))
        return 4;

    // emoticon
    if (regex_search(sub, m, EMOTICONS))
    {
        tokenize_aux(v, s, begin_index, begin_index+m.position(1));
        v.push_back(m[1].str());
        if (!trim(m[2].str()).empty()) v.push_back(m[2].str());
        tokenize_aux(v, s, begin_index+m.position(2)+m[2].str().size(), end_index);
        return 5;
    }

    return 0;
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

/**
 * Returns the beginning index of the source string that matches the target string.
 * If there is no match, return string::npos.
 */
size_t find(string source, string target, size_t source_begin, size_t source_end)
{
    bool found;
    
    for (size_t i=source_begin; i+target.size()<=source_end; i++)
    {
        found = true;

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

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
#include <iostream>
using namespace std;


// ======================================== Constants ========================================

/** Network protocols (e.g., http://). */
const std::regex NETWORK_PROTOCOL(
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
const std::regex EMOTICON(
        R"((\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(\s|[\!\.\?\,\;]|$))");

/**
 * jinho@elit.com
 * jinho.choi@elit.com
 * choi@elit.emory.edu
 * jinho:choi@0.0.0.0
 */
const std::regex EMAIL(
        R"([A-Za-z0-9\-\._]+(:\S+)?@(([A-Za-z0-9\-]+\.)+[A-Za-z]{2,3}|\d{1,3}(\.\d{1,3}){3}))");

/** &arrow; &#123; */
const std::regex HTML_ENTITY(
        R"(\&([A-Za-z]+|\#[Xx]?\d+)\;)");

/** #happy2017,@JinhoChoi */
const std::regex HASHTAG(
        R"((^|\W)([\#\@][A-Za-z][A-Za-z0-9_]+))");


// ======================================== Regular Expression ========================================

/** Considers the entire matching string as one token. */
bool tokenize_regex_aux(string s, size_t begin, size_t end, vector<string> &v, const regex &r)
{
    auto it = sregex_iterator(s.begin()+begin, s.begin()+end, r);

    if (it != sregex_iterator())
    {
        auto m = *it;
        auto tok = m[0].str();
        auto idx = begin + m.position(0);

        tokenize(s, begin, idx, v);
        v.push_back(tok);
        tokenize(s, idx + tok.size(), end, v);
        return true;
    }

    return false;
}

/** Tokenizes hyperlinks. */
bool tokenize_regex_hyperlink(string s, size_t begin, size_t end, vector<string> &v, const regex &r)
{
    auto it = sregex_iterator(s.begin()+begin, s.begin()+end, r);

    if (it != sregex_iterator())
    {
        auto m = *it;

        if (m.position(0) > 0)
        {
            auto idx = begin + m.position(0);
            tokenize(s, begin, idx, v);
            v.push_back(substr(s, idx, end));
        }
        else
            v.push_back(substr(s, begin, end));

        return true;
    }

    return false;
}

bool tokenize_regex_emoticon(string s, size_t begin, size_t end, vector<string> &v, const regex &r)
{
    auto it = sregex_iterator(s.begin()+begin, s.begin()+end, r);

    if (it != sregex_iterator())
    {
        cout << "EMO: " << substr(s, begin, end) << endl;
        auto m = *it;

        tokenize(s, begin, begin+m.position(1), v);
        v.push_back(m[1].str());
        if (!trim(m[2].str()).empty()) v.push_back(m[2].str());
        tokenize(s, begin+m.position(2)+m[2].str().size(), end, v);

        return true;
    }

    return false;
}

/** Tokenizes using regular expressions. */
int tokenize_regex(string s, size_t begin, size_t end, vector<string> &v)
{
    smatch m;

    // html entity: "&larr;", "&#8592;", "&#x2190;"
    if (tokenize_regex_aux(s, begin, end, v, HTML_ENTITY))
        return 1;

    // email: "id@elit.com", "id:pw@elit.emory.edu", "id@0.0.0.0"
    if (tokenize_regex_aux(s, begin, end, v, EMAIL))
        return 2;

    // hyperlink: "http://...", "sftp://..."
    if (tokenize_regex_hyperlink(s, begin, end, v, NETWORK_PROTOCOL))
        return 3;

    // hashtag: "#ELIT", "@ELIT"
    if (tokenize_regex_aux(s, begin, end, v, HASHTAG))
        return 4;

    // emoticon: ":-)", ":P"
    if (tokenize_regex_emoticon(s, begin, end, v, EMOTICON))
        return 5;

    return 0;
}






// ======================================== Tokenization ========================================

/**
 * Tokenizes the input string into linguistic tokens and saves them into a vector.
 * @param s the input string.
 * @return a vector of tokens from the input string; if there is no valid token, an empty vector is returned.
 */
vector<string> tokenize(string s)
{
    size_t begin = 0, end = 1;
    vector<string> v;
    s = trim(s);

    for (; end<s.size(); end++)
    {
        if (isspace(s[end]) != 0)
        {
            tokenize(s, begin, end, v);
            begin = end + 1;
        }
    }

    tokenize(s, begin, end, v);
    return v;
}

/**
 * Tokenizes s[begin:end] into linguistic tokens and adds them to the vector.
 * @param s the input string.
 * @param begin the beginning index of the string to be tokenized (inclusive).
 * @param end the ending index of the string to be tokenized (exclusive).
 * @param v the vector where tokens to be added.
 * @return non-zero if any token is added; otherwise, 0.
 */
int tokenize(std::string s, size_t begin, size_t end, std::vector<std::string> &v)
{
    // out of range
    if (end > s.size() || begin >= end)
        return 0;

    if (tokenize_regex(s, begin, end, v) != 0)
        return 1;

//    string lower = tolower(s, begin, end);
//
//    if (PRESERVE.find(lower) != PRESERVE.end())
//    {
//        v.push_back(substr(s, begin, end));
//        return true;
//    }

    v.push_back(substr(s, begin, end));
    return 10;
}
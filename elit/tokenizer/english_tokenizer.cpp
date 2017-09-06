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
#include "english_tokenizer.hpp"
#include "string_utils.hpp"
#include "io_utils.hpp"
#include "global_const.h"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

// ======================================== Constants ========================================

/** Locale: UTF8. */
const locale LOC_UTF8("en_US.UTF-8");

/** Network protocols (e.g., http://). */
const wregex RE_PROTOCOL(L"(([[:alpha:]]{3,})(:\\/\\/))");

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
const wregex RE_EMOTICON(L"(\\:\\w+\\:|\\<[\\/\\\\]?3|[\\(\\)\\\\|\\*\\$][\\-\\^]?[\\:\\;\\=]|[\\:\\;\\=B8][\\-\\^]?[3DOPp\\@\\$\\*\\\\\\)\\(\\/\\|])(\\W|$)");

/**
 * jinho@elit.com
 * jinho.choi@elit.com
 * choi@elit.emory.edu
 * jinho:choi@0.0.0.0
 */
const wregex RE_EMAIL(L"([[:alnum:]\\-\\._]+(:\\S+)?@(([[:alnum:]\\-]+\\.)+[[:alpha:]]{2,3}|\\d{1,3}(\\.\\d{1,3}){3}))");

/** &arrow; &#123; */
const wregex RE_HTML_ENTITY(L"(\\&([[:alpha:]]+|\\#[Xx]?\\d+)\\;)");

/** [1], (1a), [A.1] */
const wregex RE_LIST_ITEM(L"(([\\[\\(]+)(\\d+[[:alpha:]]?|[[:alpha:]]\\d*|\\W+)(\\.(\\d+|[[:alpha:]]))*([\\]\\)])+)");

/** doesn't */
const wregex RE_APOSTROPHE(L"[[:alpha:]](n['\u2019]t|['\u2019](ll|nt|re|ve|[dmstz]))(\\W|$)", regex_constants::icase);

/** a.b.c */
const wregex RE_ABBREVIATION(L"^[[:alpha:]](\\.[[:alpha:]])*");

const set<wstring> SET_UNIT = read_word_set(RESOURCE_ROOT + "units.txt");

const set<wstring> SET_ABBREVIATION_PERIOD = read_word_set(RESOURCE_ROOT + "abbreviation_period.txt");

const set<wstring> SET_HYPHEN_PREFIX = read_word_set(RESOURCE_ROOT + "english_hyphen_prefix.txt");

const set<wstring> SET_HYPHEN_SUFFIX = read_word_set(RESOURCE_ROOT + "english_hyphen_suffix.txt");

const map<wstring,vector<size_t>> MAP_CONCAT_WORD = read_concat_word_map(RESOURCE_ROOT + "english_concat_words.txt");

// ======================================== Tokenization ========================================

t_vector tokenize(wstring s)
{
    size_t begin, end;
    t_vector v;

    // skip preceding spaces
    for (begin=0; begin<s.size(); begin++)
        if (!isspace(s[begin], LOC_UTF8))
            break;

    for (end=begin+1; end<s.size(); end++)
    {
        if (isspace(s[end], LOC_UTF8))
        {
            tokenize_aux(v, s, begin, end);
            begin = end + 1;
        }
    }

    tokenize_aux(v, s, begin, end);
    return v;
}

namespace py = pybind11;

PYBIND11_MODULE(english_tokenizer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("tokenize", &tokenize, "A function which adds two numbers");
}

/** This is recursively called by many other functions. */
bool tokenize_aux(t_vector &v, wstring s, size_t begin, size_t end)
{
    if (begin >= end || end > s.size())
        return false;

    if (tokenize_trivial(v, s, begin, end))
        return true;

    if (tokenize_regex(v, s, begin, end))
        return true;

    if (tokenize_symbol(v, s, begin, end))
        return true;

    add_token_sub(v, s, begin, end);
    return true;
}

/** Tokenizes general cases. */
bool tokenize_trivial(t_vector &v, wstring s, size_t begin, size_t end)
{
    // single character
    if (end - begin == 1)
    {
        add_token_sub(v, s, begin, end);
        return true;
    }

    // contains only alphabets and digits
    if (all_of(s.begin()+begin, s.begin()+end, ::isalnum))
    {
        add_token_sub(v, s, begin, end);
        return true;
    }

    return false;
}

// ======================================== Add Token ========================================

void add_token(t_vector &v, wstring token, size_t begin, size_t end)
{
    if (!add_token_merge(v, token, begin, end) && !add_token_split(v, token, begin, end))
        v.emplace_back(token, make_pair(begin, end));
}

void add_token_sub(t_vector &v, wstring s, size_t begin, size_t end)
{
    add_token(v, substr(move(s), begin, end), begin, end);
}

bool add_token_merge(t_vector &v, wstring token, size_t begin, size_t end)
{
    if (v.size() >= 2)
    {
        auto prev = v[v.size()-2].first;
        auto curr = v[v.size()-1].first;
        auto next = token;

        if (curr.size() == 1)
        {
            bool merge = false;
            auto c = curr[0];

            if ((c == '&' || c == '|' || c == '/'))
            {
                merge = (prev.size() == 1 && next.size() == 1) ||
                        (all_of(next.begin(), next.end(), ::isupper) && all_of(next.begin(), next.end(), ::isupper));
            }
            else if (is_hyphen(c))
            {
                if ((prev.size() == 1 && next.size() == 1) ||
                    (all_of(next.begin(), next.end(), ::isupper) && all_of(next.begin(), next.end(), ::isupper)))
                {
                    merge = true;
                }
                else
                {
                    boost::to_lower(prev);
                    boost::to_lower(next);
                    merge = (SET_HYPHEN_PREFIX.count(prev) > 0 && all_of(next.begin(), next.end(), ::isalnum)) ||
                            (SET_HYPHEN_SUFFIX.count(next) > 0 && all_of(prev.begin(), prev.end(), ::isalnum));
                }
            }
            else if (c == '.')
            {
                boost::to_lower(prev);

                if (regex_match(prev, RE_ABBREVIATION) ||
                    SET_ABBREVIATION_PERIOD.count(prev) > 0 ||
                    (boost::iequals(prev, L"no") && isdigit(next[0], LOC_UTF8)))
                {
                    auto p1 = v[v.size()-1];
                    auto p2 = v[v.size()-2];
                    v.erase(v.end()-2, v.end());
                    v.emplace_back(p2.first + p1.first, make_pair(p2.second.first, p1.second.second));
                }
            }

            if (merge)
            {
                auto p1 = v[v.size()-1];
                auto p2 = v[v.size()-2];
                v.erase(v.end()-2, v.end());
                v.emplace_back(p2.first + p1.first + token, make_pair(p2.second.first, end));
                return true;
            }
        }
    }

    return false;
}

bool add_token_split(t_vector &v, wstring token, size_t begin, size_t end)
{
    return add_token_split_unit(v, token, begin, end) || add_token_split_concat(v, token, begin, end);
}

bool add_token_split_unit(t_vector &v, wstring token, size_t begin, size_t end)
{
    for (int i=token.size()-1; i>=0; i--)
    {
        if (isdigit(token[i], LOC_UTF8))
        {
            if (++i >= token.size()) break;
            auto t = substr(token, i, token.size());

            if (SET_UNIT.count(boost::to_lower_copy(t)) > 0)
            {
                v.emplace_back(substr(token, 0, i), make_pair(begin, begin+i));
                v.emplace_back(t, make_pair(begin+i, end));
                return true;
            }

            break;
        }
    }

    return false;
}

bool add_token_split_concat(t_vector &v, wstring token, size_t begin, size_t end)
{
    wstring t = boost::to_lower_copy(token);
    auto it = MAP_CONCAT_WORD.find(t);

    if (it != MAP_CONCAT_WORD.end())
    {
        size_t curr = 0;

        for (auto last : it->second)
        {
            v.emplace_back(substr(token, curr, last), make_pair(begin+curr, begin+last));
            curr = last;
        }

        return true;
    }

    return false;
}

// ======================================== Regular Expression ========================================

/** Tokenizes using regular expressions. */
bool tokenize_regex(t_vector &v, wstring s, size_t begin, size_t end)
{
    // html entity: "&larr;", "&#8592;", "&#x2190;"
    if (tokenize_regex_aux(v, s, begin, end, RE_HTML_ENTITY, regex_group))
        return true;

    // email: "id@elit.com", "id:pw@elit.emory.edu", "id@0.0.0.0"
    if (tokenize_regex_aux(v, s, begin, end, RE_EMAIL, regex_group))
        return true;

    // hyperlink: "http://...", "sftp://..."
    if (tokenize_regex_aux(v, s, begin, end, RE_PROTOCOL, regex_hyperlink))
        return true;

    // emoticon: ":-)", ":P"
    if (tokenize_regex_aux(v, s, begin, end, RE_EMOTICON, regex_group, 1))
        return true;

    // list time: "[1]", "(1a)", "(A.1)"
    if (tokenize_regex_aux(v, s, begin, end, RE_LIST_ITEM, regex_group))
        return true;

    // apostrophe: "does n't", "he's"
    if (tokenize_regex_aux(v, s, begin, end, RE_APOSTROPHE, regex_group, 1))
        return true;

    return false;
}

bool tokenize_regex_aux(t_vector &v, wstring s, size_t begin, size_t end, wregex r, regex_aux f, size_t flag)
{
    auto it = wsregex_iterator(s.begin()+begin, s.begin()+end, r);

    if (it != wsregex_iterator())
    {
        f(v, s, begin, end, *it, flag);
        return true;
    }

    return false;
}

/** Considers the entire matching string as one token. */
void regex_group(t_vector &v, wstring s, size_t begin, size_t end, wsmatch m, size_t flag)
{
    auto tok = m[flag].str();
    auto idx = begin + m.position(flag);
    auto lst = idx + tok.size();

    tokenize_aux(v, s, begin, idx);
    add_token(v, tok, idx, lst);
    tokenize_aux(v, s, lst, end);
}

/** Tokenizes hyperlinks. */
void regex_hyperlink(t_vector &v, wstring s, size_t begin, size_t end, wsmatch m, size_t flag)
{
    if (m.position(0) > 0)
    {
        auto idx = begin + m.position(0);
        tokenize_aux(v, s, begin, idx);
        add_token_sub(v, s, idx, end);
    }
    else
        add_token_sub(v, s, begin, end);
}

// ======================================== Symbol ========================================

bool tokenize_symbol(t_vector &v, wstring s, size_t begin, size_t end)
{
    for (auto curr=begin; curr<end; curr++)
    {
        if (skip_symbol(s, begin, end, curr))
            continue;

        if (tokenize_symbol(v, s, begin, end, curr, is_separator, tokenize_symbol_true))
            return true;

        if (tokenize_symbol(v, s, begin, end, curr, is_symbol_edge, tokenize_symbol_edge))
            return true;

        if (tokenize_symbol(v, s, begin, end, curr, is_currency_like, tokenize_symbol_currency_like))
            return true;
    }

    return false;
}

bool skip_symbol(wstring s, size_t begin, size_t end, size_t curr)
{
    auto c = s[curr];

    // .1, +1
    if (c == '.' || c == '+')
        return curr+1 < end && isdigit(s[curr+1], LOC_UTF8);

    // 1,000,000
    if (c == ',')
        return curr-1 >= begin && isdigit(s[curr-1], LOC_UTF8) &&
               curr+3 < end && isdigit(s[curr+1], LOC_UTF8) && isdigit(s[curr+2], LOC_UTF8) && isdigit(s[curr+3], LOC_UTF8) &&
               (curr+4 >= end || !isdigit(s[curr+4], LOC_UTF8));

    // '97
    if (is_single_quote(c))
        return curr+2 < end && isdigit(s[curr+1], LOC_UTF8) && isdigit(s[curr+2], LOC_UTF8) &&
               (curr+3 >= end || !isdigit(s[curr+3], LOC_UTF8));

    if (c == ':')
        return curr-1 >= begin && isdigit(s[curr-1], LOC_UTF8) &&
               curr+1 < end && isdigit(s[curr+1], LOC_UTF8);

    return false;
}

//bool skip_symbol_period(wstring s, size_t begin, size_t end, size_t curr)
//{
//    if (s[curr] != '.') return false;
//    wstring sub = substr(s, begin, curr);
//
//    // a.b.c.
//    if (regex_match(sub, RE_ABBREVIATION))
//        return true;
//
//    // ph.d.
//    boost::to_lower(sub, LOC_UTF8);
//    return SET_ABBREVIATION_PERIOD.count(sub) > 0;
//}

bool tokenize_symbol(t_vector &v, wstring s, size_t begin, size_t end, size_t curr, symbol_aux_0 f0, symbol_aux_1 f1)
{
    if (f0(s[curr]))
    {
        auto last = get_last_sequence_index(s, curr, end);

        if (f1(s, begin, end, curr, last))
        {
            tokenize_aux(v, s, begin, curr);
            add_token_sub(v, s, curr, last);
            tokenize_aux(v, s, last, end);
            return true;
        }
    }

    return false;
}

bool tokenize_symbol_true(wstring s, size_t begin, size_t end, size_t curr, size_t last)
{
    return true;
}

bool tokenize_symbol_edge(wstring s, size_t begin, size_t end, size_t curr, size_t last)
{
    return curr+1 < last || curr == begin || last == end || ispunct(s[curr-1], LOC_UTF8) || ispunct(s[last], LOC_UTF8);
}

bool tokenize_symbol_currency_like(wstring s, size_t begin, size_t end, size_t curr, size_t last)
{
    return curr+1 < last || last == end || isdigit(s[last], LOC_UTF8);
}

bool is_separator(wchar_t c)
{
    return c == ',' || c == ';' || c == ':' || c == '~' || c == '&' || c == '|' || c == '/' || is_bracket(c) || is_arrow(c) || is_double_quote(c) || is_hyphen(c);
}

bool is_symbol_edge(wchar_t c)
{
    return is_single_quote(c) || is_final_mark(c);
}

bool is_currency_like(wchar_t c)
{
    return c == '#' || is_currency(c);
}

size_t get_last_sequence_index(wstring s, size_t curr, size_t end)
{
    auto c = s[curr];
    auto final_mark = is_final_mark(c);
    size_t last;

    for (last=curr+1; last<end; last++)
    {
        auto d = s[last];

        if (final_mark)
        {
            if (!is_final_mark(d))
                break;
        }
        else if (c != d)
            break;

    }

    return last;
}
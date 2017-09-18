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
#include "../string_utils.hpp"
#include "../io_utils.hpp"
#include "../global_const.h"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

// ======================================== Lexicons ========================================

/** Locale: UTF8. */
const locale LOC_UTF8("en_US.UTF-8");

const string ROOT = RESOURCE_ROOT + "/tokenizer/";

const set<wstring> SET_ABBREVIATION_PERIOD = read_word_set(ROOT + "english_abbreviation_period.txt");
const set<wstring> SET_APOSTROPHE_FRONT = read_word_set(ROOT + "english_apostrophe_front.txt");
const map<wstring,vector<size_t>> MAP_CONCAT_WORD = read_concat_word_map(ROOT + "english_concat_words.txt");
const set<wstring> SET_HYPHEN_PREFIX = read_word_set(ROOT + "english_hyphen_prefix.txt");
const set<wstring> SET_HYPHEN_SUFFIX = read_word_set(ROOT + "english_hyphen_suffix.txt");
const set<wstring> SET_UNIT = read_word_set(ROOT + "units.txt");

// ======================================== Regular Expressions ========================================

/** Network protocols (e.g., http://). */
const wregex RE_PROTOCOL(L"((http|https|ftp|sftp|ssh|ssl|telnet|smtp|pop3|imap|imap4|sip)(:\\/\\/))");

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
const wregex RE_EMOTICON(L"(\\:\\w+\\:|\\<[\\/\\\\]?3|[\\(\\)\\\\|\\*\\$][\\-\\^]?[\\:\\;\\=]|[\\:\\;\\=B8]([\\-\\^]+)?[3DOPp\\@\\$\\*\\\\\\)\\(\\/\\|]+)(\\W|$)");

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
const wregex RE_LIST_ITEM(L"(([\\[\\(\\{\\<]+)(\\d+[[:alpha:]]?|[[:alpha:]]\\d*|\\W+)(\\.(\\d+|[[:alpha:]]))*([\\]\\)\\}\\>])+)");

/** doesn't */
const wregex RE_APOSTROPHE(L"[[:alpha:]](n['\u2019]t|['\u2019](ll|nt|re|ve|[dmstz]))(\\W|$)", regex_constants::icase);

/** a.b.c */
const wregex RE_ABBREVIATION(L"^[[:alnum:]]([\\.-][[:alnum:]])*");

// ======================================== Tokenization ========================================

TokenList tokenize(wstring s)
{
    size_t begin, end;
    TokenList v;

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

bool tokenize_aux(TokenList &v, wstring s, size_t begin, size_t end)
{
    if (begin >= end || end > s.size())
        return false;

    if (tokenize_trivial(v, s, begin, end))
        return true;

    if (tokenize_regex(v, s, begin, end))
        return true;

    if (tokenize_symbol(v, s, begin, end))
        return true;

    add_token_aux(v, s, begin, end);
    return true;
}

bool tokenize_trivial(TokenList &v, wstring s, size_t begin, size_t end)
{
    if (end - begin == 1 || all_of(s.begin()+begin, s.begin()+end, ::isalnum))
    {
        add_token_aux(v, s, begin, end);
        return true;
    }

    return false;
}

// ======================================== Add Token ========================================

void add_token(TokenList &v, wstring token, size_t begin, size_t end)
{
    if (!concat_token(v, token, begin, end) && !split_token(v, token, begin, end))
        v.emplace_back(token, begin, end);
}

void add_token_aux(TokenList &v, wstring s, size_t begin, size_t end)
{
    add_token(v, substr(move(s), begin, end), begin, end);
}

// ======================================== Concatenate ========================================

bool concat_token(TokenList &v, wstring token, size_t begin, size_t end)
{
    if (!v.empty())
    {
        auto p = v.back();
        const auto &prev = get_form(p);
        const auto &curr = token;

        if (is_concat_apostrophe_front(prev, curr) || is_concat_abbreviation(prev, curr))
        {
            v.pop_back();
            v.emplace_back(prev+curr, get_begin(p), end);
            return true;
        }
    }

    if (v.size() >= 2)
    {
        auto p = v[v.size()-2];
        auto c = v[v.size()-1];
        const auto &prev = get_form(p);
        const auto &curr = get_form(c);
        const auto &next = token;

        if (is_concat_acronym(prev, curr, next) || is_concat_hyphen(prev, curr, next))
        {
            v.erase(v.end()-2, v.end());
            v.emplace_back(prev+curr+next, get_begin(p), end);
            return true;
        }

        concat_token_no(v, prev, curr, next);
    }

    return false;
}

bool is_concat_apostrophe_front(wstring prev, wstring curr)
{
    if (prev.size() == 1 && is_single_quote(prev[0]))
    {
        boost::to_lower(curr);
        return SET_APOSTROPHE_FRONT.count(curr) > 0;
    }

    return false;
}

bool is_concat_abbreviation(wstring prev, wstring curr)
{
    if (curr == L".")
    {
        boost::to_lower(prev);
        return regex_match(prev, RE_ABBREVIATION) || SET_ABBREVIATION_PERIOD.count(prev) > 0;
    }

    return false;
}

bool is_concat_acronym(wstring prev, wstring curr, wstring next)
{
    auto c = curr[0];

    if (curr.size() == 1 && (c == '&' || c == '|' || c == '/'))
    {
        return (prev.size() <= 2 && next.size() <= 2) ||
               (all_of(prev.begin(), prev.end(), ::isupper) && all_of(next.begin(), next.end(), ::isupper));
    }

    return false;
}

bool is_concat_hyphen(wstring prev, wstring curr, wstring next)
{
    if (curr.size() == 1 && is_hyphen(curr[0]))
    {
        // 000-0000, 000-000-000
        if (3 <= prev.size() && all_of(prev.end()-3, prev.end(), ::isdigit) && (prev.size() == 3 || is_hyphen(prev[prev.size()-4])) &&
            3 <= next.size() && next.size() <= 4 && all_of(next.begin(), next.end(), ::isdigit))
            return true;

        // p-u-s-h
        if (isalnum(prev.back(), LOC_UTF8) && (prev.size() == 1 || is_hyphen(prev[prev.size()-2])) &&
            next.size() == 1 && isalnum(next[0], LOC_UTF8))
            return true;

        boost::to_lower(prev);
        boost::to_lower(next);
        return (SET_HYPHEN_PREFIX.count(prev) > 0 && all_of(next.begin(), next.end(), ::isalnum)) ||
               (SET_HYPHEN_SUFFIX.count(next) > 0 && all_of(prev.begin(), prev.end(), ::isalnum));
    }

    return false;
}

bool concat_token_no(TokenList &v, wstring prev, wstring curr, wstring next)
{
    if (boost::iequals(prev, L"no") && curr == L"." && isdigit(next[0], LOC_UTF8))
    {
        auto p = v[v.size()-2];
        auto c = v[v.size()-1];
        v.erase(v.end()-2, v.end());
        v.emplace_back(prev+curr, get_begin(p), get_end(c));
        return true;
    }

    return false;
}

// ======================================== Split ========================================

bool split_token(TokenList &v, wstring token, size_t begin, size_t end)
{
    return split_token_unit(v, token, begin, end) || split_token_concat_words(v, token, begin, end) || split_token_final_mark(v, token, begin, end);
}

bool split_token_unit(TokenList &v, wstring token, size_t begin, size_t end)
{
    for (int i=token.size()-1; i>=0; i--)
    {
        if (isdigit(token[i], LOC_UTF8))
        {
            if (++i >= token.size()) break;
            auto t = substr(token, i, token.size());

            if (SET_UNIT.count(boost::to_lower_copy(t)) > 0)
            {
                v.emplace_back(substr(token, 0, i), begin, begin+i);
                v.emplace_back(t, begin+i, end);
                return true;
            }

            break;
        }
    }

    return false;
}

bool split_token_concat_words(TokenList &v, wstring token, size_t begin, size_t end)
{
    wstring t = boost::to_lower_copy(token);
    auto it = MAP_CONCAT_WORD.find(t);

    if (it != MAP_CONCAT_WORD.end())
    {
        size_t curr = 0;

        for (auto last : it->second)
        {
            v.emplace_back(substr(token, curr, last), begin+curr, begin+last);
            curr = last;
        }

        return true;
    }

    return false;
}

#include <iostream>
bool split_token_final_mark(TokenList &v, wstring token, size_t begin, size_t end)
{
    if (token.size() < 9)
        return false;

    for (int i=3; i<token.size()-4; i++)
    {
        if (is_final_mark(token[i]) && all_of(token.begin(), token.begin()+i, ::isalpha) && all_of(token.begin()+i+1, token.end(), ::isalpha))
        {
            v.emplace_back(substr(token, 0, i), begin, begin+i);
            v.emplace_back(wstring(1, token[i]), begin+i, begin+i+1);
            v.emplace_back(substr(token, i+1, token.size()), begin+i+1, end);
            return true;
        }
    }

    return false;
}

// ======================================== Regular Expression ========================================

/** Tokenizes using regular expressions. */
bool tokenize_regex(TokenList &v, wstring s, size_t begin, size_t end)
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

bool tokenize_regex_aux(TokenList &v, wstring s, size_t begin, size_t end, wregex r, regex_aux f, size_t flag)
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
void regex_group(TokenList &v, wstring s, size_t begin, size_t end, wsmatch m, size_t flag)
{
    auto tok = m[flag].str();
    auto idx = begin + m.position(flag);
    auto lst = idx + tok.size();

    tokenize_aux(v, s, begin, idx);
    add_token(v, tok, idx, lst);
    tokenize_aux(v, s, lst, end);
}

/** Tokenizes hyperlinks. */
void regex_hyperlink(TokenList &v, wstring s, size_t begin, size_t end, wsmatch m, size_t flag)
{
    if (m.position(0) > 0)
    {
        auto idx = begin + m.position(0);
        tokenize_aux(v, s, begin, idx);
        add_token_aux(v, s, idx, end);
    }
    else
        add_token_aux(v, s, begin, end);
}

// ======================================== Symbol ========================================

bool tokenize_symbol(TokenList &v, wstring s, size_t begin, size_t end)
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

    // -1
    if (c == '-')
        return curr == begin && curr+1 < end && isdigit(s[curr+1], LOC_UTF8);

    // 1,000,000
    if (c == ',')
        return curr-1 >= begin && isdigit(s[curr-1], LOC_UTF8) &&
               curr+3 < end && isdigit(s[curr+1], LOC_UTF8) && isdigit(s[curr+2], LOC_UTF8) && isdigit(s[curr+3], LOC_UTF8) &&
               (curr+4 >= end || !isdigit(s[curr+4], LOC_UTF8));

    // '97
    if (is_single_quote(c))
        return curr+2 < end && isdigit(s[curr+1], LOC_UTF8) && isdigit(s[curr+2], LOC_UTF8) &&
               (curr+3 >= end || !isdigit(s[curr+3], LOC_UTF8));

    // 10:30
    if (c == ':')
        return curr-1 >= begin && isdigit(s[curr-1], LOC_UTF8) &&
               curr+1 < end && isdigit(s[curr+1], LOC_UTF8);

    return false;
}

bool tokenize_symbol(TokenList &v, wstring s, size_t begin, size_t end, size_t curr, symbol_aux_0 f0, symbol_aux_1 f1)
{
    if (f0(s[curr]))
    {
        auto last = get_last_sequence_index(s, curr, end);

        if (f1(s, begin, end, curr, last))
        {
            tokenize_aux(v, s, begin, curr);
            add_token_aux(v, s, curr, last);
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
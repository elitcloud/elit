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

#define CATCH_CONFIG_RUNNER

#include <iostream>
#include <fstream>
#include "english_tokenizer.hpp"
#include "../catch.hpp"

using Catch::Matchers::VectorContains;
using namespace std;

void print_tokens(wstring s)
{
    for (auto t : tokenize(s))
        wcout << get_form(t) << ' ' << get_begin(t) << ' ' << get_end(t) << endl;
}

TEST_CASE("White Spaces")
{
    SECTION("empty string")
    {
        wstring s;
        TokenList gold;
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("only white space")
    {
        wstring s = L" \n\t";
        TokenList gold;
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("no space")
    {
        wstring s = L"ABC";
        TokenList gold = {make_tuple(L"ABC", 0, 3)};
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("trim")
    {
        wstring s = L" A\tBC  D12 \n";

        TokenList gold = {
                make_tuple(  L"A", 1,  2),
                make_tuple( L"BC", 3,  5),
                make_tuple(L"D12", 7, 10)};

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Concatenate")
{
    SECTION("apostrophe front")
    {
        wstring s = L"'CAUSE 'tis ' em";

        TokenList gold = {
                make_tuple(L"'CAUSE",  0,  6),
                make_tuple(L"'tis"  ,  7, 11),
                make_tuple( L"'em"  , 12, 16)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("abbreviation")
    {
        wstring s = L"A.B. a.B.c AB.C. Ph.D.";

        TokenList gold = {
                make_tuple(L"A.B." ,  0,  4),
                make_tuple(L"a.B.c",  5, 10),
                make_tuple(L"AB.C" , 11, 15),
                make_tuple(L"."    , 15, 16),
                make_tuple(L"Ph.D.", 17, 22)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("acronym")
    {
        wstring s = L"ab&cd AB|CD 1/2";

        TokenList gold = {
                make_tuple(L"ab"   ,  0,  2),
                make_tuple(L"&"    ,  2,  3),
                make_tuple(L"cd"   ,  3,  5),
                make_tuple(L"AB|CD",  6, 11),
                make_tuple(L"1/2"  , 12, 15)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("hyphenated")
    {
        wstring s = L"mis-predict mic-predict book-able book-es";

        TokenList gold = {
                make_tuple(L"mis-predict",  0, 11),
                make_tuple(L"mic"        , 12, 15),
                make_tuple(L"-"          , 15, 16),
                make_tuple(L"predict"    , 16, 23),
                make_tuple(L"book-able"  , 24, 33),
                make_tuple(L"book"       , 34, 38),
                make_tuple(L"-"          , 38, 39),
                make_tuple(L"es"         , 39, 41)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("no.")
    {
        wstring s = L"No. 1 No. a No.";

        TokenList gold = {
                make_tuple(L"No.",  0, 3),
                make_tuple(L"1"  ,  4, 5),
                make_tuple(L"No" ,  6, 8),
                make_tuple(L"."  ,  8, 9),
                make_tuple(L"a"  , 10, 11),
                make_tuple(L"No" , 12, 14),
                make_tuple(L"."  , 14, 15)};

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Split")
{
    SECTION("unit")
    {
        wstring s = L"20mg 100cm 1st 11a.m. 10PM";

        TokenList gold = {
                make_tuple(L"20"  ,  0,  2),
                make_tuple(L"mg"  ,  2,  4),
                make_tuple(L"100" ,  5,  8),
                make_tuple(L"cm"  ,  8, 10),
                make_tuple(L"1st" , 11, 14),
                make_tuple(L"11"  , 15, 17),
                make_tuple(L"a.m.", 17, 21),
                make_tuple(L"10"  , 22, 24),
                make_tuple(L"PM"  , 24, 26)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("concatenated words")
    {
        wstring s = L"whadya DON'CHA";

        TokenList gold = {
                make_tuple(L"wha",  0,  3),
                make_tuple(L"d"  ,  3,  4),
                make_tuple(L"ya" ,  4,  6),
                make_tuple(L"DO" ,  7,  9),
                make_tuple(L"N'" ,  9, 11),
                make_tuple(L"CHA", 11, 14)};

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Regular Expression")
{
    SECTION("html entity")
    {
        wstring s = L"ab&larr;cd&#8592;&#x2190;ef&rarr;";

        TokenList gold = {
                make_tuple(L"ab"      ,  0,  2),
                make_tuple(L"&larr;"  ,  2,  8),
                make_tuple(L"cd"      ,  8, 10),
                make_tuple(L"&#8592;" , 10, 17),
                make_tuple(L"&#x2190;", 17, 25),
                make_tuple(L"ef"      , 25, 27),
                make_tuple(L"&rarr;"  , 27, 33)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("email")
    {
        wstring s = L"a;jinho@elit.com,b;jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0";

        TokenList gold = {
                make_tuple(L"a"                  ,  0,  1),
                make_tuple(L";"                  ,  1,  2),
                make_tuple(L"jinho@elit.com"     ,  2, 16),
                make_tuple(L","                  , 16, 17),
                make_tuple(L"b"                  , 17, 18),
                make_tuple(L";"                  , 18, 19),
                make_tuple(L"jinho.choi@elit.com", 19, 38),
                make_tuple(L","                  , 38, 39),
                make_tuple(L"choi@elit.emory.edu", 39, 58),
                make_tuple(L","                  , 58, 59),
                make_tuple(L"jinho:choi@0.0.0.0" , 59, 77)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("html entity")
    {
        wstring s = L"ab&larr;cd&#8592;&#x2190;ef&rarr;";

        TokenList gold = {
                make_tuple(L"ab"      ,  0,  2),
                make_tuple(L"&larr;"  ,  2,  8),
                make_tuple(L"cd"      ,  8, 10),
                make_tuple(L"&#8592;" , 10, 17),
                make_tuple(L"&#x2190;", 17, 25),
                make_tuple(L"ef"      , 25, 27),
                make_tuple(L"&rarr;"  , 27, 33)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("html entity")
    {
        wstring s = L"ab&larr;cd&#8592;&#x2190;ef&rarr;";

        TokenList gold = {
                make_tuple(L"ab"      ,  0,  2),
                make_tuple(L"&larr;"  ,  2,  8),
                make_tuple(L"cd"      ,  8, 10),
                make_tuple(L"&#8592;" , 10, 17),
                make_tuple(L"&#x2190;", 17, 25),
                make_tuple(L"ef"      , 25, 27),
                make_tuple(L"&rarr;"  , 27, 33)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("network protocol")
    {
        wstring s = L"a:http://ab sftp://ef abeftp://";

        TokenList gold = {
                make_tuple(L"a"        ,  0,  1),
                make_tuple(L":"        ,  1,  2),
                make_tuple(L"http://ab",  2, 11),
                make_tuple(L"sftp://ef", 12, 21),
                make_tuple(L"abe"      , 22, 25),
                make_tuple(L"ftp://"   , 25, 31)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("emoticon")
    {
        wstring s = L":-) A:-( :). B:smile::sad: C:):(! :)., :)";

        TokenList gold = {
                make_tuple(L":-)"    ,  0,  3),
                make_tuple(L"A"      ,  4,  5),
                make_tuple(L":-("    ,  5,  8),
                make_tuple(L":)"     ,  9, 11),
                make_tuple(L"."      , 11, 12),
                make_tuple(L"B"      , 13, 14),
                make_tuple(L":smile:", 14, 21),
                make_tuple(L":sad:"  , 21, 26),
                make_tuple(L"C"      , 27, 28),
                make_tuple(L":)"     , 28, 30),
                make_tuple(L":("     , 30, 32),
                make_tuple(L"!"      , 32, 33),
                make_tuple(L":)"     , 34, 36),
                make_tuple(L"."      , 36, 37),
                make_tuple(L","      , 37, 38),
                make_tuple(L":)"     , 39, 41),

        };

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("list")
    {
        wstring s = L"[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]";

        TokenList gold = {
                make_tuple(L"[a]"         ,  0,  3),
                make_tuple(L"(1)"         ,  3,  6),
                make_tuple(L"(1.a)"       ,  6, 11),
                make_tuple(L"[11.22.a.33]", 11, 23),
                make_tuple(L"(A.1)"       , 23, 28),
                make_tuple(L"[a1]"        , 28, 32),
                make_tuple(L"["           , 32, 33),
                make_tuple(L"hello"       , 33, 38),
                make_tuple(L"]"           , 38, 39),
        };

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("apostrophe")
    {
        wstring s = L"don't he's can't does'nt 0's DON'T ab'cd";

        TokenList gold = {
                make_tuple(L"do"    ,  0,  2),
                make_tuple(L"n't"   ,  2,  5),
                make_tuple(L"he"    ,  6,  8),
                make_tuple(L"'s"    ,  8, 10),
                make_tuple(L"ca"    , 11, 13),
                make_tuple(L"n't"   , 13, 16),
                make_tuple(L"does"  , 17, 21),
                make_tuple(L"'nt"   , 21, 24),
                make_tuple(L"0's"   , 25, 28),
                make_tuple(L"DO"    , 29, 31),
                make_tuple(L"N'T"   , 31, 34),
                make_tuple(L"ab'cd" , 35, 40),
        };

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Misc")
{
    wstring s = L"Mbaaah!hello prize.Please";
    print_tokens(s);

    s = L"|http://www.clearnlp.com www.clearnlp.com |mailto:support@clearnlp.com|jinho_choi@clearnlp.com|";
    print_tokens(s);

    s = L":-))) :---( Hi:).";
    print_tokens(s);

    s = L"---\"((``@#$Choi%&*''))\".?!===";
    print_tokens(s);

    s = L",,A---C**D~~~~E==F,G,,H..I.J-1.--2-K||L-#3";
    print_tokens(s);

    s = L"(e.g., bcd. BCD. and. T. T.. T.";
    print_tokens(s);

    s = L"$1 E2 L3 USD1 2KPW $1 USD1 us$ US$ ub$";
    print_tokens(s);

    s = L"I did it my way. Definitely not worth stopping by.";
    print_tokens(s);
}





//TEST_CASE("Compound nouns with dashes are tokenized together", "[compounds]"){
//    CHECK(tokenize(L"I really need a pick-me-up!").size() == 6);
//    CHECK(tokenize(L"I don't like hand-me-down clothes!").size() == 7);
//}

//TEST_CASE("Novel -- Check range sanity", "[novel]")
//{
//    wifstream fin("/home/catherine/dev/elit/elit/cpp/test_data/wiki_text.txt");
//    wstring line;
//    while(getline(fin, line)){
//        auto tokens = tokenize(line);
//        SECTION("Checking range correspondence with the original text string")
//        {
//            for (auto token : tokens)
//            {
//                size_t start_pos = get_begin(token);
//                size_t length = token.second.second - start_pos;
//                wstring expected_string = line.substr(start_pos, length);
//                CHECK(expected_string == token.first);
//            }
//        }
//
//        SECTION("Checking ranges for illegal overlaps")
//        {
//            for (int i = 0; i < tokens.size(); ++i)
//            {
//
//                auto token = tokens[i];
//
//                pair<size_t, size_t> curr = token.second;
//                CHECK(curr.second > curr.first);
//                if (i > 0)
//                {
//                    pair<size_t, size_t> prev = tokens[i-1].second;
//                    CHECK(curr.first >= prev.second);
//                    if (curr.first < prev.second){
//                        wcout << line << endl;
//                        wcout << "LENGTH: " << line.size() << endl;
//                        wcout << "PREV: " << line.substr(prev.first, prev.second - prev.first) << endl;
//                        wcout << "CURR: " << line.substr(curr.first, curr.second - curr.first) << endl;
//                        wcout << "TOKENS:" << endl;
//                        for (auto token : tokens){
//                            wcout << token.first << " ";
//                        }
//                        wcout << endl;
//                    }
//                }
//
//                CHECK(curr.second <= line.length());
//            }
//        }
//    }
//}

int main(int argc, const char *const *const argv)
{
    Catch::Session().run(argc, argv);
}

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
        wstring s = L"A.B. a.B.c AB.C. Ph.D. 1.2. A-1.";

        TokenList gold = {
                make_tuple(L"A.B." ,  0,  4),
                make_tuple(L"a.B.c",  5, 10),
                make_tuple(L"AB.C" , 11, 15),
                make_tuple(L"."    , 15, 16),
                make_tuple(L"Ph.D.", 17, 22),
                make_tuple(L"1.2." , 23, 27),
                make_tuple(L"A-1." , 28, 32)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("acronym")
    {
        wstring s = L"ab&cd AB|CD 1/2 a-1 1-2";

        TokenList gold = {
                make_tuple(L"ab&cd",  0,  5),
                make_tuple(L"AB|CD",  6, 11),
                make_tuple(L"1/2"  , 12, 15),
                make_tuple(L"a-1"  , 16, 19),
                make_tuple(L"1-2"  , 20, 23)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("hyphenated")
    {
        wstring s = L"mis-predict mic-predict book - able book-es 000-0000 000-000-000 p-u-s-h-1-2";

        TokenList gold = {
                make_tuple(L"mis-predict",  0, 11),
                make_tuple(L"mic"        , 12, 15),
                make_tuple(L"-"          , 15, 16),
                make_tuple(L"predict"    , 16, 23),
                make_tuple(L"book-able"  , 24, 35),
                make_tuple(L"book"       , 36, 40),
                make_tuple(L"-"          , 40, 41),
                make_tuple(L"es"         , 41, 43),
                make_tuple(L"000-0000"   , 44, 52),
                make_tuple(L"000-000-000", 53, 64),
                make_tuple(L"p-u-s-h-1-2", 65, 76)};

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

    SECTION("final mark")
    {
        wstring s = L"Mbaaah.Please hello!world";

        TokenList gold = {
                make_tuple(L"Mbaaah",  0,  6),
                make_tuple(L"."     ,  6,  7),
                make_tuple(L"Please",  7, 13),
                make_tuple(L"hello" , 14, 19),
                make_tuple(L"!"     , 19, 20),
                make_tuple(L"world" , 20, 25)};

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
        wstring s = L":-) A:-( :). B:smile::sad: C:):(! :)., :-))) :---( Hi:).";

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
                make_tuple(L":-)))"  , 39, 44),
                make_tuple(L":---("  , 45, 50),
                make_tuple(L"Hi"     , 51, 53),
                make_tuple(L":)"     , 53, 55),
                make_tuple(L"."      , 55, 56)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("list")
    {
        wstring s = L"[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]{22}";

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
                make_tuple(L"{22}"        , 39, 43)};

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
                make_tuple(L"ab'cd" , 35, 40)};

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Symbol")
{
    SECTION("skip")
    {
        wstring s = L".1 +1 -1 1,000,000.00 1,00 '97 '90s '1990 10:30 1:2 a:b";

        TokenList gold = {
                make_tuple(L".1"          ,  0,  2),
                make_tuple(L"+1"          ,  3,  5),
                make_tuple(L"-1"          ,  6,  8),
                make_tuple(L"1,000,000.00",  9, 21),
                make_tuple(L"1"           , 22, 23),
                make_tuple(L","           , 23, 24),
                make_tuple(L"00"          , 24, 26),
                make_tuple(L"'97"         , 27, 30),
                make_tuple(L"'90s"        , 31, 35),
                make_tuple(L"'"           , 36, 37),
                make_tuple(L"1990"        , 37, 41),
                make_tuple(L"10:30"       , 42, 47),
                make_tuple(L"1:2"         , 48, 51),
                make_tuple(L"a"           , 52, 53),
                make_tuple(L":"           , 53, 54),
                make_tuple(L"b"           , 54, 55)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("hyphenated digit")
    {
        wstring s = L"1997-2012,1990's-2000S '97-2012 '12-14's";

        TokenList gold = {
                make_tuple(L"1997"  ,  0,  4),
                make_tuple(L"-"     ,  4,  5),
                make_tuple(L"2012"  ,  5,  9),
                make_tuple(L","     ,  9, 10),
                make_tuple(L"1990's", 10, 16),
                make_tuple(L"-"     , 16, 17),
                make_tuple(L"2000S" , 17, 22),
                make_tuple(L"'97"   , 23, 26),
                make_tuple(L"-"     , 26, 27),
                make_tuple(L"2012"  , 27, 31),
                make_tuple(L"'12"   , 32, 35),
                make_tuple(L"-"     , 35, 36),
                make_tuple(L"14's"  , 36, 40)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("separator")
    {
        wstring s = L"aa;;;b:c\"\"d\"\" 0'''s ''a''a'' 'a'a'a'.?!..";

        TokenList gold = {
                make_tuple(L"aa"   ,  0,  2),
                make_tuple(L";;;"  ,  2,  5),
                make_tuple(L"b"    ,  5,  6),
                make_tuple(L":"    ,  6,  7),
                make_tuple(L"c"    ,  7,  8),
                make_tuple(L"\"\"" ,  8, 10),
                make_tuple(L"d"    , 10, 11),
                make_tuple(L"\"\"" , 11, 13),
                make_tuple(L"0"    , 14, 15),
                make_tuple(L"'''"  , 15, 18),
                make_tuple(L"s"    , 18, 19),
                make_tuple(L"''"   , 20, 22),
                make_tuple(L"a"    , 22, 23),
                make_tuple(L"''"   , 23, 25),
                make_tuple(L"a"    , 25, 26),
                make_tuple(L"''"   , 26, 28),
                make_tuple(L"'"    , 29, 30),
                make_tuple(L"a'a'a", 30, 35),
                make_tuple(L"'"    , 35, 36),
                make_tuple(L".?!..", 36, 41)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("hashtag")
    {
        wstring s = L"#happy2018,@Jinho_Choi: ab@cde";

        TokenList gold = {
                make_tuple(L"#happy2018" ,  0, 10),
                make_tuple(L","          , 10, 11),
                make_tuple(L"@Jinho_Choi", 11, 22),
                make_tuple(L":"          , 22, 23),
                make_tuple(L"ab@cde"     , 24, 30)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("currency")
    {
        wstring s = L"#1 $1 ";

        TokenList gold = {
                make_tuple(L"#", 0, 1),
                make_tuple(L"1", 1, 2),
                make_tuple(L"$", 3, 4),
                make_tuple(L"1", 4, 5)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("segment")
    {
        wstring s = L"Hello world! \"I'm Jinho.\" \"Dr. Choi\" (I'm a prof...) [[Really?]] Yes!";
        vector<int> b = segment(tokenize(s));
        int g[] = {0, 3, 9, 20, 24, 26};

        for (int i=0; i<b.size(); i++)
            CHECK(g[i] == b[i]);
    }
}

TEST_CASE("Benchmark")
{
//    for (TokenList tokens: segment(contents))
//    {
//        for (Token token : tokens) wcout << get_form(token) << ' ';
//        wcout << endl;
//    }

//    wifstream fin("/Users/jdchoi/workspace/elit/sample.txt");
//    wstring contents{istreambuf_iterator<wchar_t>(fin), istreambuf_iterator<wchar_t>()};
//
//    long long tt = 0, wc = 0;
//    TokenList ls;
//    wstring line;
//
//    while(getline(fin, line))
//    {
//        auto st = chrono::high_resolution_clock::now();
//        ls = tokenize(line);
//        auto et = chrono::high_resolution_clock::now();
//        wc += ls.size();
//        tt += chrono::duration_cast<std::chrono::milliseconds>(et-st).count();
//
//        for (Token t : ls) wcout << get_form(t) << ' ';
//        wcout << endl;
//    }
//
//    cout << wc * 1000 / tt << endl;
}

int main(int argc, const char *const *const argv)
{
    init("/Users/jdchoi/workspace/elit/resources/tokenizer/");
//    init("./elit/resources/tokenizer/");
    Catch::Session().run(argc, argv);
}

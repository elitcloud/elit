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

//#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER

#include <iostream>
#include <fstream>
#include "english_tokenizer.hpp"
#include "../catch.hpp"

using Catch::Matchers::VectorContains;
using namespace std;

TEST_CASE("White Spaces")
{
    SECTION("empty string")
    {
        wstring s;
        TokenList gold;
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("contains only white space")
    {
        wstring s = L" \n\t";
        TokenList gold;
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("no space exists")
    {
        wstring s = L"ABC";
        TokenList gold = {make_tuple(L"ABC", 0, 3)};
        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("trim spaces")
    {
        wstring s = L" A\tBC  D12 \n";

        TokenList gold = {
                make_tuple(  L"A", 1,  2),
                make_tuple( L"BC", 3,  5),
                make_tuple(L"D12", 7, 10)};

        CHECK(equals(gold, tokenize(s)));
    }
}

TEST_CASE("Apostrophe Front")
{
    SECTION("'cause")
    {
        wstring s = L"'CAUSE I'm the best";

        TokenList gold = {
                make_tuple(L"'CAUSE",  0,  6),
                make_tuple(     L"I",  7,  8),
                make_tuple(    L"'m",  8, 10),
                make_tuple(   L"the", 11, 14),
                make_tuple(  L"best", 15, 19)};

        CHECK(equals(gold, tokenize(s)));
    }

    SECTION("'tis")
    {
        wstring s = L"There 'tis again!";

        TokenList gold = {
                make_tuple(L"There",  0,  5),
                make_tuple( L"'tis",  6, 10),
                make_tuple(L"again", 11, 16),
                make_tuple(    L"!", 16, 17)};

        CHECK(equals(gold, tokenize(s)));
    }
}



//TEST_CASE("Contractions with apostrophes are tokenized together", "[them]"){
//    REQUIRE(get_form(tokenize(L"Hit 'em hard")[1]) == L"'em");
//    CHECK(tokenize(L"There 'tis again!").size() == 4);
//    CHECK(tokenize(L"Did Mr. Lapham say 'twas wrong?").size() == 7);
//}
//
//TEST_CASE("Compound nouns with dashes are tokenized together", "[compounds]"){
//    CHECK(tokenize(L"I really need a pick-me-up!").size() == 6);
//    CHECK(tokenize(L"I don't like hand-me-down clothes!").size() == 7);
//}
//TEST_CASE("Dummy", "[dummy]"){
//    REQUIRE(true);
//}

//TEST_CASE("Testing on Wikipedia text", "[wiki]"){
//    CHECK(wiki_test().first);
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
    int result = Catch::Session().run(argc, argv);
}

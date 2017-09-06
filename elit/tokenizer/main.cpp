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
#include "english_tokenizer.hpp"
using namespace std;

void print_v(wstring s)
{
    t_vector v = tokenize(s);
    wcout << s << " -> ";
    for (auto t : v) wcout << t.first << ' ';
    wcout << "\n";
}

int main(int argc, const char * argv[])
{
    wstring t[] = {
            L"",
            L"  ",
            L"AB CD EF",
            L"  AB CD EF",
            L"AB CD EF  ",
            L"  AB CD EF  ",
            L"AB  CD  EF",
            L"ab&larr;cd&#8592;&#x2190;ef&#X2190;",
            L"http://ab cd.ftp://ef eftp",
            L"jinho@elit.com,jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0",
            L"#happy2018,@Jinho_Choi: ab@cde",
            L":-) A:-( :). B:smile::sad: C:):(! :)., :)",
            L"[a](1)(1.a)[11.22.a.33](A.1)[a1]",
            L"1997-2012,1990's-2000S '97-2012 '12-14's '90s '900",
            L"1997,1998/1999/2000",
            L"aa;;;b:c\"\"d\"\"",
            L"don't he's can't does'nt 0's DON'T",
            L"0'''s ''a''a'' 'a'a'a'",
            L"a,b,c,d,",
            L"0.12 .34 +1 ",
            L"#1. $1,000-2K.?!'''...",
            L"\"w.r.t. this. Ph.D. Ph.D.\"",
            L"inter-state, far-most, inter - state, far - most",
            L"w/o ab/cd AB/CD",
            L"No. 1 No. You!",
            L"11:30pm 3:0 a:b ab:cd",
            L"20mg 100cm 11a.m. 10PM",
            L"whadya DON'CHA",
    };

    for (auto s : t) print_v(s);
    return 0;
}

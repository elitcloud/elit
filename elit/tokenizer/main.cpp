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
            L"He possesses  dollars!"
        L"It is simply papier-mâché", //unresolved!
            L"The antique clock is from c.1900!", //unresolved!
            L"Have you received my C.V.?",//unresolved
            L"Ling et al. shall not get their way!",
            L"Broccoli, etc. will not be a problem.",
            L"Ibid., 29", //fixed
            L"RC J.D. is a good man!",
            L"I need 1 lb. of food!", //fixed
            L"One equals three, Q.E.D.!", //unresolved
            L"R.I.P., Mr. Finley!"


    };

    for (auto s : t) print_v(s);
    return 0;
}

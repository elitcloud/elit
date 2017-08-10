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


// ======================================== Utilities ========================================




















void print_v(string s)
{
    vector<string> v = tokenize(s);
    cout << "[" << s << "] -> [";
    for (string t : v) cout << t << ",";
    cout << "]\n";
}

int main(int argc, const char * argv[])
{
    string t[] = {"", "  ", "AB CD EF", "  AB CD EF", "AB CD EF  ", "  AB CD EF  ", "AB  CD  EF"};
    for (string s : t) print_v(s);
    return 0;
}

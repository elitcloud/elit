#include "tokenizer/english_tokenizer.hpp"
#include <set>

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
#include "io_utils.hpp"
#include "string_utils.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/format.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>

using namespace std;

set<wstring> read_word_set(string filename)
{
    ifstream fin(filename);
    set<wstring> s;
    string line;

//  fin.imbue(locale(std::locale(), new codecvt_utf8<wchar_t>));

    while (getline(fin, line))
    {
        boost::trim(line);
        if (!line.empty()) s.insert(utf8_to_wstring(line));
    }

    cout << boost::format("Init: %s (%d keys)\n") % filename % s.size();
    return s;
}

map<wstring,vector<size_t>> read_concat_word_map(string filename)
{
    map<wstring,vector<size_t>> m;
    ifstream fin(filename);
    string line;

//  fin.imbue(locale(std::locale(), new codecvt_utf8<wchar_t>));

    while (getline(fin, line))
    {
        boost::trim(line);
        if (!line.empty())
        {
            vector<size_t> v;
            wstringstream k;
            size_t i = 0;

            for (wchar_t c : utf8_to_wstring(line))
            {
                if (c == ' ')
                    v.push_back(i);
                else
                {
                    k << c;
                    i++;
                }
            }

            v.push_back(i);
            m[k.str()] = v;
        }
    }

    cout << boost::format("Init: %s (%d keys)\n") % filename % m.size();
    return m;
}
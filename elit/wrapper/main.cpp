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
#include "tokenizer.h"
#include "util/string_util.hpp"

using namespace std;

// ======================================== Tokenization ========================================

vector<string> tokenize(string s)
{
    int begin_index = 0, end_index = 1;
    vector<string> v;
    char c;

    s = trim(s);
    
    for (; end_index<s.size(); end_index++)
    {
        c = s[end_index];
        
        if (isspace(c))
        {
            append(v, s, begin_index, end_index);
            begin_index = end_index + 1;
        }
    }
    
    append(v, s, begin_index, end_index);
    return v;
}

/**
 * Peforms v.append(s[begin_index:end_index]);
 * Returns true if s[begin_index:end_index] is valid; otherwise, false.
 */
bool append(vector<string> &v, string s, int begin_index, int end_index)
{
    if (begin_index >= 0 && end_index <= s.size() && begin_index < end_index)
    {
        s = s.substr(begin_index, end_index - begin_index);
        v.push_back(s);
        return true;
    }
    
    return false;
}

void tokenize_hyperlink(string s, int begin_index, int end_index)
{
    for (int i=begin_index; i<end_index; i++)
    {
        
    }
    
}


// ======================================== Utilities ========================================




















void print_v(vector<string> v)
{
    cout << v.size() << ":";
    for (string s : v) cout << s << ",";
    cout << "\n";
}

void test_string_util()
{
    tokenize("");
    tokenize("  ");
    tokenize("AB CD EF");
    tokenize("  AB CD EF");
    tokenize("AB CD EF  ");
    tokenize("  AB CD EF  ");
    tokenize("AB  CD  EF");
}


int main(int argc, const char * argv[])
{
    //    string s = "my name is Jinho D. Choi.";
    //    vector<string> v = tokenize(s);
    //    for (string s : v)
    //        cout << s << "\n";
    
    test_string_util();
    
    return 0;
}

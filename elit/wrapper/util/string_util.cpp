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

#include "string_util.hpp"
using namespace std;

/** Returns a string where all beginning and ending white spaces are trimmed from the specific string. */
string trim(string s)
{
    unsigned long idx;

    // trim beginning white spaces
    for (idx=0; idx<s.size(); idx++)
    {
        if (!isspace(s[idx]))
        {
            if (idx > 0) s.erase(0, idx);
            break;
        }
    }
    
    // contain only white spaces
    if (idx == s.size())
        return "";
    
    unsigned long lst = s.size() - 1;

    // trim endding white spaces
    for (idx=lst; idx>0; idx--)
    {
        if (!isspace(s[idx]))
        {
            if (idx < lst) s.erase(idx+1, lst-idx);
            break;
        }
    }
        
    return s;
}

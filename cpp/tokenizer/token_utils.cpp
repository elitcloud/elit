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

#include "token_utils.hpp"
#include <iostream>

using namespace std;

// ======================================== Token ========================================

Token create_token(wstring token, size_t begin, size_t end)
{
    return make_tuple(token, begin, end);
}

wstring get_form(Token t)
{
    return get<0>(t);
}

size_t get_begin(Token t)
{
    return get<1>(t);
}

size_t get_end(Token t)
{
    return get<2>(t);
}

bool equals(Token t1, Token t2)
{
    return get_form(t1) == get_form(t2) && get_begin(t1) == get_begin(t2) && get_end(t1) == get_end(t2);
}

bool equals(TokenList t1, TokenList t2)
{
    if (t1.size() != t2.size())
        return false;

    for (int i=t1.size()-1; i>=0; i--)
    {
        if (!equals(t1[i], t2[i]))
            return false;
    }

    return true;
}
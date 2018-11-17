# ========================================================================
# Copyright 2018 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
from elit.tools import EnglishTokenizer, SpaceTokenizer
tok = EnglishTokenizer()
text = "Mr. Johnson doesn't like cats! What's his favorite then? He likes puffy-dogs."
print(tok.decode(text))

# tok = EnglishTokenizer()
#
# # email address
# text = 'Email (support@elit.cloud)'
# print(tok.decode(text))
#
# # hyperlink
# text = 'URL: https://elit.cloud'
# print(tok.decode(text))
#
# # emoticon
# text = 'I love ELIT :-)!?.'
# print(tok.decode(text))
#
# # hashtag
# text = 'ELIT is the #1 platform #elit2018.'
# print(tok.decode(text))
#
# # html entity
# text = 'A&larr;B'
# print(tok.decode(text))
#
# # hyphens for telephone/social security number
# text = '(123) 456-7890, 123-456-7890, 2014-2018'
# print(tok.decode(text))
#
# # list item
# text = '(A)First (A.1)Second [2a]Third [Forth]'
# print(tok.decode(text))
#
# # units
# text = "$1,000 20mg 100cm 11:00a.m. 10:30PM"
# print(tok.decode(text))
#
# # acronym
# text = "I'm gonna miss Dr. Choi 'cause he isn't here."
# print(tok.decode(text))
#

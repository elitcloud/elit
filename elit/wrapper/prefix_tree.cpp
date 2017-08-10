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
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
//#include "prefix_tree.hpp"

using namespace std;

class Node
{
public:
    Node();
    Node(string key);
    ~Node();
    
    /** Return the child with the specific key if exists; otherwise, insert a child with the key and return the new child. */
    Node* insert(string key);
    
private:
    string key;
    vector<string> values;
    unordered_map<string, Node*> children;
};

Node::Node()
{
    key = "";
}

Node::Node(string key)
{
    this->key = key;
}

Node::~Node()
{
    // TODO: delete all nodes in the subtree
}

Node* Node::insert(string key)
{
    unordered_map<string, Node*>::iterator it = children.find(key);
    
    if (it != children.end())
        return it->second;
    else
    {
        Node* node = new Node(key);
        children[key] = node;
        return node;
    }
}







class PrefixTree
{
public:
    PrefixTree();
    ~PrefixTree();
    
    /** Insert the key chain to the tree, where the chain can include multiple keys delimited by white spaces (' '). */
    void insert(string key_chain);
    
private:
    Node* root;
};

PrefixTree::PrefixTree()
{
    root = new Node();
}

PrefixTree::~PrefixTree()
{
    delete root;
}

void PrefixTree::insert(string key_chain)
{
    int begin_index = 0, end_index = 1;
    Node* node = root;
    string key;
    
    for (; end_index<key_chain.size(); end_index++)
    {
        if (isspace(key_chain[end_index]))
        {
            key = key_chain.substr(begin_index, end_index - begin_index);
            node = node->insert(key);
            
            
            
            
            begin_index = end_index + 1;
        }
    }
    
    if (begin_index < end_index && end_index <= key.size())
        key = key_chain.substr(begin_index, end_index - begin_index);
}

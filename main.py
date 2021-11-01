#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         tamp-fundation-algorithm
# Author:       wendi
# Date:         2021/10/19
from TreeExercise import TrieTree

if __name__ == '__main__':
    trie = TrieTree()
    print(trie.insert("apple"))
    print(trie.insert("fdd"))
    print(trie.search("apple"))
    print(trie.search("app"))
    print(trie.startsWith("app"))
    print(trie.insert("app"))
    print(trie.search("app"))




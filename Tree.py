#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         Tree
# Author:       wendi
# Date:         2021/10/30


class TrieTree(object):
    def __init__(self):
        super(TrieTree, self).__init__()
        self.children = [None] * 26
        self.isEnd = False

    def searchPrefix(self, prefix: str) -> 'TrieTree':
        node = self
        for ch in prefix:
            idx = ord(ch) - ord('a')
            if not node.children[idx]:
                return None
            node = node.children[idx]
        return node

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            idx = ord(ch) - ord('a')
            if not node.children[idx]:
                node.children[idx] = TrieTree()
            node = node.children[idx]
        node.isEnd = True

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        return self.searchPrefix(prefix) is not None



#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         TreeExercise
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


class TrieNode(object):
    def __init__(self):
        super(TrieNode, self).__init__()
        self.child = {}
        self.flag = None


class LcpTree(object):
    def __init__(self):
        super(LcpTree, self).__init__()
        self.root = TrieNode()

    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        elif len(strs) == 1:
            return strs[0]

        # 将strs中所有字符串插入Trie树
        for words in strs:
            curNode = self.root
            for word in words:
                if curNode.child.get(word) is None:
                    curNode.child[word] = TrieNode()
                curNode = curNode.child[word]
            curNode.flag = 1

        curNode = self.root

        lcp = []
        while curNode.flag != 1:
            # 遍历Trie树，直至当前节点的子节点分叉数大于1
            if len(curNode.child) == 1:
                curNodeChar = list(curNode.child.keys())[0]
                lcp.append(curNodeChar)
                curNode = curNode.child[curNodeChar]
            else:
                break
        return ''.join(lcp)

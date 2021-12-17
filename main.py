#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         tamp-fundation-algorithm
# Author:       wendi
# Date:         2021/10/19
from TreeExercise import TrieTree

from typing import List


class Combination:
    def solution(self, arr: List[str], k: int) -> List[List[int]]:
        def backtrack(start, arr, k):
            if len(path) == k:
                res.append(path[:])
                return

            for i in range(start, len(arr) + 1):
                if i <= len(arr) - k + len(path) + 1:
                    path.append(arr[i-1])
                    backtrack(i + 1, arr, k)
                    path.pop()

        res = []
        path = []
        backtrack(1, arr, k)
        return res


if __name__ == '__main__':
    # trie = TrieTree()
    # print(trie.insert("apple"))
    # print(trie.insert("fdd"))
    # print(trie.search("apple"))
    # print(trie.search("app"))
    # print(trie.startsWith("app"))
    # print(trie.insert("app"))
    # print(trie.search("app"))

    obj = Combination()
    res = obj.solution(arr=['a', 'b', 'c', 'd'], k=2)
    print(res)




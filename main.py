#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         tamp-fundation-algorithm
# Author:       wendi
# Date:         2021/10/19
# from TreeExercise import TrieTree

# from typing import List
import sys
import urllib2
import json

# class Combination:
#     def solution(self, arr: List[str], k: int) -> List[List[int]]:
#         def backtrack(start, arr, k):
#             if len(path) == k:
#                 res.append(path[:])
#                 return
#
#             for i in range(start, len(arr) + 1):
#                 if i <= len(arr) - k + len(path) + 1:
#                     path.append(arr[i-1])
#                     backtrack(i + 1, arr, k)
#                     path.pop()
#
#         res = []
#         path = []
#         backtrack(1, arr, k)
#         return res
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # trie = TrieTree()
    # print(trie.insert("apple"))
    # print(trie.insert("fdd"))
    # print(trie.search("apple"))
    # print(trie.search("app"))
    # print(trie.startsWith("app"))
    # print(trie.insert("app"))
    # print(trie.search("app"))

    # obj = Combination()
    # res = obj.solution(arr=['a', 'b', 'c', 'd'], k=2)
    # print(res)

    # url = 'https://restapi.amap.com/v3/geocode/geo?output=JSON&key=c39f275eb61e7b1499963c74475920d0'
    req_url = 'https://restapi.amap.com/v3/staticmap?location=116.481485,39.990464&zoom=10&size=750*300&markers=mid,,A:116.481485,39.990464&key=c39f275eb61e7b1499963c74475920d0'

    res_data = urllib2.urlopen(urllib2.Request(req_url))
    with open('test.png', 'wb') as file:
        file.write(res_data.fp.read())
    print(type(res_data))
    # location = json.loads(res_data.read())
    # print(res_data.read())






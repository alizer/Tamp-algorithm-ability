#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         UnionFindAlgo
# Author:       wendi
# Date:         2022/5/19

#  并查集系列算法
from typing import List


class Node:
    def __init__(self, v):
        self.value = v


class UnionFind:
    """
    并查集基础算法
    """
    def __init__(self, values: List):
        self.nodes = dict()
        self.parents = dict()
        self.size_map = dict()
        for cur in values:
            node = Node(cur)
            self.nodes[cur] = node
            self.parents[node] = node
            self.size_map[node] = 1

    def findFather(self, cur: Node):
        """
        给你一个节点，请你往上到不能再往上，把代表节点返回
        :param cur:
        :return:
        """
        path_stack = []
        while cur != self.parents.get(cur):
            path_stack.append(cur)
            cur = self.parents.get(cur)

        while path_stack:
            self.parents[path_stack.pop()] = cur

        return cur

    def isSameSet(self, a, b):
        return self.findFather(self.nodes.get(a)) == self.findFather(self.nodes.get(b))

    def union(self, a, b):
        a_head = self.findFather(self.nodes.get(a))
        b_head = self.findFather(self.nodes.get(b))

        if a_head != b_head:
            a_set_size = self.size_map.get(a_head)
            b_set_size = self.size_map.get(b_head)
            big = a_head if a_set_size >= b_set_size else b_head
            small = b_head if big == a_head else a_head
            self.parents[small] = big
            self.size_map[big] = a_set_size + b_set_size
            self.size_map.pop(small)

        return self.size_map.__len__()


class MergeUsers:
    """
    并查集应用，合并用户
    如果两个user，a字段一样、或者b字段一样、或者c字段一样，就认为是一个人
    请合并users，返回合并之后的用户数量
    """

    class User:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    def mergeUsers(self, users: List[User]):
        pass


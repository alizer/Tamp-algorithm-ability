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

    def getSetNum(self):
        return len(self.size_map)


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
        unionfind = UnionFind(values=users)
        map_a = dict()
        map_b = dict()
        map_c = dict()

        for user in users:
            if user.a in map_a.keys():
                unionfind.union(user, map_a.get(user.a))
            else:
                map_a[user.a] = user

            if user.b in map_b.keys():
                unionfind.union(user, map_b.get(user.b))
            else:
                map_b[user.b] = user

            if user.c in map_c.keys():
                unionfind.union(user, map_c.get(user.c))
            else:
                map_c[user.c] = user

        return unionfind.getSetNum()


if __name__ == '__main__':
    obj = MergeUsers()
    o1 = obj.User(1, 3, 8)
    o2 = obj.User(2, 3, 6)
    o3 = obj.User(9, 0, 7)
    o4 = obj.User(91, 1, 0)
    res = obj.mergeUsers([o1, o2, o3, o4])
    print(res)




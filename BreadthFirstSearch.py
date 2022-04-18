#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         BreadthFirstSearch
# Author:       wendi
# Date:         2022/3/4

from collections import deque
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CBTInserter:
    """
    剑指 Offer II 043. 往完全二叉树添加节点
    https://leetcode-cn.com/problems/NaqhDT/
    """

    def __init__(self, root: TreeNode):
        self.root = root
        self.nodequeue = deque()
        stack = deque([root])
        while stack:
            cur_node = stack.popleft()
            if not cur_node.left or not cur_node.right:
                self.nodequeue.append(cur_node)
            if cur_node.left:
                stack.append(cur_node.left)
            if cur_node.right:
                stack.append(cur_node.right)

    def insert(self, v: int) -> int:
        self.nodequeue.append(TreeNode(v))
        if self.nodequeue[0].left:
            self.nodequeue[0].right = self.nodequeue[-1]
            return self.nodequeue.popleft().val
        else:
            self.nodequeue[0].left = self.nodequeue[-1]
            return self.nodequeue[0].val

    def get_root(self) -> TreeNode:
        return self.root


class LargestValues:
    """
    https://leetcode-cn.com/problems/hPov7L/
    剑指 Offer II 044. 二叉树每层的最大值
    """
    def solution(self, root: TreeNode) -> List[int]:
        """
        用两个队列实现二叉树的广度优先搜索
        把不同层的节点放入到不同的队列中
        队列queue1只放当前遍历层的节点
        队列queue2只放下一层的节点
        :param root:
        :return:
        """
        # queue1存放当前遍历层的节点
        queue1 = []
        # queue2存放下一层的节点
        queue2 = []
        if root:
            queue1.append(root)
        res = []
        max_value = float('-inf')
        while queue1:
            node = queue1.pop(0)
            max_value = max(max_value, node.val)
            if node.left:
                queue2.append(node.left)
            if node.right:
                queue2.append(node.right)

            # 当queue1队列被清空时，当前层的所有节点都已经被遍历完了
            if not queue1:
                res.append(max_value)
                max_value = float('-inf')

                # 开始遍历下一层之前，重新赋值
                queue1 = queue2
                queue2 = []

        return res

    def solution1(self, root: TreeNode) -> List[int]:
        """
        用两个变量来判断当前层还是下一层
        1, 用两个变量来分别记录两层节点的数量
        2, current记录当前层中位于队列之中节点的数量
        3, next记录下一层中位于队列之中节点的数量
        :param root:
        :return:
        """
        current, next = 0, 0
        queue = []
        res = []
        max_value = float('-inf')
        if root:
            queue.append(root)
            current += 1

        while queue:
            node = queue.pop(0)
            current -= 1
            max_value = max(max_value, node.val)
            if node.left:
                queue.append(node.left)
                next += 1
            if node.right:
                queue.append(node.right)
                next += 1

            if current == 0:
                res.append(max_value)
                max_value = float('-inf')
                current = next
                next = 0
        return res

    def solution2(self, root: TreeNode) -> List[int]:
        """
        广度优先搜索，逐层遍历
        :param root:
        :return:
        """
        if not root:
            return []
        queue = [root]
        ret = []
        while queue:
            cur_size = len(queue)
            max_val = float('-inf')
            for i in range(cur_size):
                cur = queue.pop(0)
                if cur.val > max_val:
                    max_val = cur.val
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            ret.append(max_val)
        return ret





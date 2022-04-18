#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         DepthFirstSearch
# Author:       wendi
# Date:         2022/3/4
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class LargestValues:
    """
    https://leetcode-cn.com/problems/hPov7L/
    剑指 Offer II 044. 二叉树每层的最大值
    """
    def solution(self, root: TreeNode) -> List[int]:
        res = []

        def dfs(root, depth):
            if not root:
                return

            if len(res) == depth:
                res.append(root.val)
            else:
                res[depth] = max(res[depth], root.val)
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
        dfs(root, 0)
        return res






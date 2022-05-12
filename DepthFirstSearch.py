#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         DepthFirstSearch
# Author:       wendi
# Date:         2022/3/4
import math
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def preorderTraversal(self, root):
        """
        前序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

    def inorderTraversal(self, root):
        """
        中序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    def postorderTraversal(self, root):
        """
        后序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

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


class IncreasingBST:
    """
    剑指 Offer II 052. 展平二叉搜索树
    https://leetcode-cn.com/problems/NYBBNL/
    """
    def solution(self, root: TreeNode) -> TreeNode:
        """
        1、先对输入的二叉搜索树执行中序遍历，将结果保存到一个列表arr中；
        2、然后根据列表中的节点值，创建等价的只含有右节点的二叉搜索树，
        其过程等价于根据节点值创建一个链表。
        :param root:
        :return:
        """
        arr = []
        def inorderTraversal(root, res):
            if not root:
                return
            inorderTraversal(root.left, res)
            res.append(root.val)
            inorderTraversal(root.right, res)
        inorderTraversal(root, arr)

        dummyNode = TreeNode(-1)
        currNode = dummyNode
        for val in arr:
            currNode.right = TreeNode(val)
            currNode = currNode.right
        return dummyNode.right

    def solution1(self, root: TreeNode) -> TreeNode:
        """
        在中序遍历的过程中改变节点指向
        :param root:
        :return:
        """
        def inorder(node, currNode):
            if not node:
                return
            inorder(node.left, currNode)

            # 在中序遍历的过程中修改节点指向
            currNode.right = node
            node.left = None
            currNode = node

            inorder(node.left, currNode)

        dummyNode = TreeNode(-1)
        resNode = dummyNode
        inorder(root, resNode)
        return dummyNode.right



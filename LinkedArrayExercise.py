#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         LinkedArrayExercise
# Author:       wendi
# Date:         2021/11/5


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MergeTwoLinkedArray(object):
    """
    https://leetcode-cn.com/problems/merge-two-sorted-lists/submissions/
    21. 合并两个有序链表
    """
    def solution(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        ***迭代***
        时间复杂度：O(n + m)，其中 n 和 m 分别为两个链表的长度
        空间复杂度：O(1)
        :param l1:
        :param l2:
        :return:
        """
        res = ListNode(-1)
        pre = res
        while l1 and l2:
            if l1.val >= l2.val:
                pre.next = l2
                l2 = l2.next
            else:
                pre.next = l1
                l1 = l1.next
            pre = pre.next
        pre.next = l1 if l1 is not None else l2

        return res.next
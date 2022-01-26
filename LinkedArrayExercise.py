#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         LinkedArrayExercise
# Author:       wendi
# Date:         2021/11/5


# Definition for singly-linked list.
from typing import List, Optional


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
        # 哨兵节点
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


class MergeKLists(object):
    """
    23. 合并K个升序链表
    https://leetcode-cn.com/problems/merge-k-sorted-lists/
    """
    def solution(self, lists: List[ListNode]) -> ListNode:
        """
        优先级队列
        时间复杂度：O(n*log(k))，n 是所有链表中元素的总和，k 是链表个数。
        :param lists:
        :return:
        """
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next

        return dummy.next

    def solution2(self, lists: List[ListNode]) -> Optional[ListNode, None]:
        """
        分而治之，链表两两合并
        :param lists:
        :return:
        """
        if not lists:
            return None
        n = len(lists)
        return self.merge(lists, 0, n-1)

    def merge(self, lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left)//2

        l1 = self.merge(lists, left, mid)
        l2 = self.merge(lists, mid+1, right)
        return self.mergeTwoLists(l1, l2)

    def mergeTwoLists(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1

        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2









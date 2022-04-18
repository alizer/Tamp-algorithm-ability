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


class RemoveNthFromEnd:
    """
    剑指 Offer II 021. 删除链表的倒数第 n 个结点
    https://leetcode-cn.com/problems/SLwz0R/
    """
    def solution(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        current, length = head, 0
        while current:
            current = current.next
            length += 1
        current = dummy
        for _ in range(length-n):
            current = current.next
        current.next = current.next.next
        return dummy.next


class DetectCycle:
    """
    https://leetcode-cn.com/problems/c32eOV/
    剑指 Offer II 022. 链表中环的入口节点
    """
    def solutin(self, head: ListNode) -> ListNode:
        """
        快慢指针
        :param head:
        :return:
        """
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        if fast is None or fast.next is None:
            return None

        slow = head

        while fast != slow:
            slow = slow.next
            fast = fast.next

        return slow


class AddTwoNumbers:
    """
    剑指 Offer II 025. 链表中的两数相加
    https://leetcode-cn.com/problems/lMSNwu/
    """
    def solution(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        反转链表
        :param l1:
        :param l2:
        :return:
        """
        def revertList(listnode):
            cur, prev = listnode, None
            while cur:
                next = cur.next
                cur.next = prev
                prev = cur
                cur = next
            return prev

        revertL1 = revertList(l1)
        revertL2 = revertList(l2)

        prev = ListNode()
        res = prev
        intVal = 0
        while revertL1 or revertL2 or intVal:
            num = 0
            if revertL1:
                num += revertL1.val
                revertL1 = revertL1.next
            if revertL2:
                num += revertL2.val
                revertL2 = revertL2.next
            intVal, num = divmod(num+intVal, 10)
            prev.next = ListNode(num)
            prev = prev.next

        return revertList(res.next)

    def solution2(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        栈
        :param l1:
        :param l2:
        :return:
        """
        arr1 = []
        arr2 = []
        while l1:
            arr1.append(l1.val)
            l1 = l1.next
        while l2:
            arr2.append(l2.val)
            l2 = l2.next
        newNode = ListNode(-1)
        intVal = 0
        while arr1 or arr2 or intVal:
            sum = intVal
            sum += arr1.pop() if arr1 else 0
            sum += arr2.pop() if arr2 else 0
            intVal, num = divmod(sum, 10)
            curNode = ListNode(num)
            curNode.next = newNode.next
            newNode.next = curNode
        return newNode.next


class ReorderList:
    """
    剑指 Offer II 026. 重排链表
    https://leetcode-cn.com/problems/LGjMqU/
    """
    def solution(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return

        mid = self.middleNode(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = self.reverseList(l2)
        self.mergeList(l1, l2)

    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverseList(self, head: ListNode) -> ListNode:
        cur, prev = head, None
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev

    def mergeList(self, l1: ListNode, l2: ListNode) -> ListNode:
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         LinkedArrayExercise
# Author:       wendi
# Date:         2021/11/5


# Definition for singly-linked list.
from typing import List, Optional


class ListNode:
    """
    单向链表
    """
    def __init__(self, val=0, next= None):
        self.val = val
        self.next = next

class DoubleListNode:
    """双向链表"""
    def __init__(self, val=0, next=None, last=None):
        self.next = next
        self.last = last
        self.val = val


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


class ReverseDoubleListNode:
    def solution(self, head: DoubleListNode) -> DoubleListNode:
        pre, next = None, None
        while head:
            next = head.next
            head.next = pre
            head.last = next
            pre = head
            head = next

        return pre


class LinkedList2QueueAndStack:
    class myQueue:
        """
        用单链表实现队列
        """
        def __init__(self, head: ListNode, tail: ListNode, size: int):
            self.head = head
            self.tail = tail
            self.size = 0

        def isEmpty(self):
            return self.size == 0

        def offer(self, value: int):
            """往队列中插入值"""
            # 生成新节点
            cur = ListNode(val=value)
            if not self.tail:  # 说明此时队列为空，将头和尾指针分别指向新节点
                self.head = cur
                self.tail = cur
            else:
                self.tail.next = cur
                self.tail = cur

            self.size += 1

        def poll(self):
            """
            出列，即将头节点从队列中移出
            :return:
            """
            res = None
            if self.head:
                res = self.head.val
                self.head = self.head.next
                self.size -= 1

            # 如果此时头节点为空了，那将尾结点也置为空
            if not self.head:
                self.tail = None

            return res

    class myStack:
        """
        用单链表实现栈
        """
        def __init__(self, head: ListNode, size: int):
            self.head = head
            self.size = size

        def isEmpty(self):
            return self.size == 0

        def push(self, value: int):
            """
            压栈
            :param value:
            :return:
            """
            # 生成新节点
            cur = ListNode(val=value)
            if not self.head:
                self.head = cur
            else:
                cur.next = self.head
                self.head = cur
            self.size += 1

        def pop(self):
            """
            出栈
            :return:
            """
            res = None
            if self.head:
                res = self.head.val
                self.head = self.head.next
                self.size -= 1

            return res


class DoubleLinkedList2Deque:
    """
    双向链表实现双向队列
    """
    class myDeque:
        def __init__(self, head: DoubleListNode, tail: DoubleListNode, size: int):
            self.head = head
            self.tail = tail
            self.size = size

        def isEmpty(self):
            return self.size == 0

        def pushHead(self, value: int):
            """
            从队列头插入数据
            :param value:
            :return:
            """
            cur = DoubleListNode(value)

            if not self.head:  # 说明此时双向队列为空，将头和尾指针分别指向新节点
                self.head = cur
                self.tail = cur
            else:
                cur.next = self.head
                self.head.last = cur
                self.head = cur
            self.size += 1

        def pushTail(self, value: int):
            """
            从队列尾插入数据
            :param value:
            :return:
            """
            cur = DoubleListNode(value)

            if not self.head:  # 说明此时双向队列为空，将头和尾指针分别指向新节点
                self.head = cur
                self.tail = cur
            else:
                self.tail.next = cur
                cur.last = self.tail
                self.tail = cur

        def pollHead(self):
            """
            从队列头弹出数据
            :param value:
            :return:
            """
            res = None
            if not self.head:
                return res

            res = self.head.val
            if self.head == self.tail:
                self.head = None
                self.tail = None
            else:
                self.head = self.head.next
                self.head.last = None

            self.size -= 1
            return res


        def pollTail(self):
            """
            从队列尾弹出数据
            :param value:
            :return:
            """
            res = None
            if not self.head:
                return res

            res = self.head.val
            if self.head == self.tail:
                self.head = None
                self.tail = None
            else:
                self.tail = self.tail.last
                self.tail.next = None

            self.size -= 1
            return res


class ReverseKGroup:
    """
    K 个一组翻转链表
    """
    def solution(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def getKGroupEnd(s, k):
            k -= 1
            while k != 0 and s:
                s = s.next
                k -= 1
            return s

        def reverse(start, end):
            end = end.next
            pre, next = None, None
            cur = start
            while cur != end:
                next = cur.next
                cur.next = pre
                pre = cur
                cur = next

            start.next = end

        start = head
        end = getKGroupEnd(start, k)
        # 凑不够一组，直接返回head
        if not end:
            return head

        # 第一组凑齐了，最终返回的head指向第一组的尾节点
        head = end
        reverse(start, end)

        # 上一组的结尾节点
        lastEnd = start
        while lastEnd.next:
            start = lastEnd.next
            end = getKGroupEnd(start, k)
            if not end:
                return head
            reverse(start, end)
            lastEnd.next = end
            lastEnd = start

        return head


class IsPalindromeList:
    """
    判断单链表是否为回文串
    """
    # def __init__(self, head: ListNode):
    #     self.head

    def isPalindrome1(self, head: ListNode):
        """
        结合栈，需要额外的 O(n) 空间
        :param head:
        :return:
        """
        stack = []
        cur = head
        while cur:
            stack.append(cur)
            cur = cur.next

        while head:
            if head.val != stack.pop().val:
                return False

            head = head.next
        return True

    def isPalindrome2(self, head: ListNode):
        """
        需要额外 n/2 空间
        :param head:
        :return:
        """
        if not head or not head.next:
            return True

        # 奇数个节点，找到中点位置
        # 偶数个节点，找到下中点位置
        right = head.next
        cur = head
        while cur.next and cur.next.next:
            right = right.next
            cur = cur.next.next

        stack = []
        while right:
            stack.append(right)
            right = right.next

        while not stack:
            if stack.pop().val != head.val:
                return False
            head = head.next

        return True

    def isPalindrome2(self, head: ListNode):
        """
        只需要O(1)的空间
        :param head:
        :return:
        """
        if not head or not head.next:
            return True

        # 找到中点 或 上中点
        n1, fast = head, head
        while n1.next and fast.next.next:
            n1 = n1.next
            fast = fast.next.next

        n2 = n1.next  # n2 -> right part first node
        n1.next = None

        while n2:
            n3 = n2.next
            n2.next = n1
            n1 = n2
            n2 = n3

        n3 = n1  # n3 -> save last node
        n2 = head  # n2 -> left first node

        res = True
        while n1 and n2:
            if n1.val != n2.val:
                res = False
                break
            n1 = n1.next
            n2 = n2.next

        n1 = n3.next
        n3.next = None
        while n1:  # recover list
            n2 = n1.next
            n1.next = n3
            n3 = n1
            n1 = n2

        return res


class SmallerEqualBigger:
    """
    将一个链表根据指定数，划分为三部分，小于等于大于
    """
    def solution1(self, head: ListNode, pivot: int):
        """
        需要 额外数组 空间大小O(N), 然而按照荷兰国旗问题处理
        :param head:
        :param pivot:
        :return:
        """
        if not head:
            return head

        nodeArr = []
        cur = head
        while cur:
            cur = cur.next
            nodeArr.append(cur)

        self.arrPartition(nodeArr, pivot)
        for i in range(1, len(nodeArr)):
            nodeArr[i-1].next = nodeArr[i].next
        return nodeArr[0]

    def arrPartition(self, nodeArr, pivot):
        small, big = -1, len(nodeArr)
        index = 0
        while index != big:
            if nodeArr[index].val < pivot:
                small += 1
                nodeArr[small], nodeArr[index] = nodeArr[index], nodeArr[small]
                index += 1
            elif nodeArr[index].val == pivot:
                index += 1
            else:
                big -= 1
                nodeArr[big], nodeArr[index] = nodeArr[index], nodeArr[big]

    def solution2(self, head: ListNode, pivot: int):
        """
        找到六个节点的引用地址即可，分别是小于区域的头尾结点；等于区域的头尾结点；大于区域的头尾结点。
        :param head:
        :param pivot:
        :return:
        """
        if not head:
            return head

        sH, sT = None, None
        eH, eT = None, None
        bH, bT = None, None

        while head:
            next = head.next
            head.next = None

            if head.val < pivot:
                if not sH:
                    sH = head
                    sT = head
                else:
                    sT.next = head
                    sT = head
            elif head.val == pivot:
                if not eH:
                    eH = head
                    eT = head
                else:
                    eT.next = head
                    eT = head
            else:
                if not bH:
                    bH = head
                    bT = head
                else:
                    bT.next = head
                    bT = head

            head = next

        # 串联三部分
        # 小于区域的尾巴，连等于区域的头，等于区域的尾巴连大于区域的头
        if sT:
            sT.next = eH
            eT = sT if not eT else eT
        # 下一步，一定是需要用eT去接大于区域的头
        # 有等于区域，eT -> 等于区域的尾结点
        # 无等于区域，eT -> 小于区域的尾结点
        # eT 尽量不为空的尾巴节点
        if eT:  # 如果小于区域和等于区域，不是都没有
            eT.next = bH

        if sH:
            return sH
        elif eH:
            return eH
        else:
            return bH

class LinkedListRandom:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class CopyListWithRandom:
    """
    https://leetcode-cn.com/problems/copy-list-with-random-pointer/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
    138. 复制带随机指针的链表
    """
    def solution(self, head: LinkedListRandom):
        """
        结合hashmap，空间复杂度O(N)
        :return:
        """
        old2new = dict()
        cur = head
        while cur:
            old2new[cur] = LinkedListRandom(cur.val, cur.next, None)
            cur = cur.next

        cur = head
        while cur:
            newNode = old2new.get(cur)
            newNode.next = old2new.get(cur.next)
            newNode.random = old2new.get(cur.random)
            cur = cur.next
        return old2new.get(head)

    def solution1(self, head: LinkedListRandom):
        """
        空间复杂度 O(1)，在每个节点后接一个镜像节点
        :param head:
        :return:
        """
        if not head:
            return None

        cur = head
        # 1 -> 2 -> 3 -> null
        # 1 -> 1' -> 2 -> 2' -> 3 -> 3'
        while cur:
            cur.next = LinkedListRandom(cur.val, cur.next, None)
            cur = cur.next.next

        cur = head
        # 依次设置 1' 2' 3' random指针
        while cur:
            next = cur.next.next
            copy = cur.next
            copy.random = cur.random.next if cur.random else None
            cur = next

        # 1' 位置，要返回结果，先记下来
        res = head.next
        cur = head
        # next方向上，把新老链表分离
        while cur:
            next = cur.next.next
            copy = cur.next
            cur.next = next
            copy.next = next.next if next else None
            cur = next

        return res


class FindFirstIntersectNode:
    """
    找到两个链表的 交点
    """
    def getIntersectNode(self, head1: ListNode, head2: ListNode):
        """

        :param head1:
        :param head2:
        :return:
        """
        loop1 = self.getLoopNode(head1)
        loop2 = self.getLoopNode(head2)
        # 两个都无环
        if not loop1 and not loop2:
            return self.noloop(head1, head2)

        # 两个都有环
        if loop1 and loop2:
            return self.bothloop(head1, loop1, head2, loop2)

        # 一个有环，一个无环，这种情况下不会有交点，直接返回None
        return None

    def getLoopNode(self, head: ListNode):
        """
        找到链表第一个入环节点，如果无环，返回null
        :param head:
        :return:
        """
        if not head or not head.next or not head.next.next:
            return None

        # n1 slow；n2 fast
        slow = head.next
        fast = head.next.next
        while slow != fast:
            if not fast.next or not fast.next.next:  # 这种情况不存在环
                return None
            fast = fast.next.next
            slow = slow.next

        # slow fast 相遇
        fast = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow

    def noloop(self, head1, head2):
        """
        如果两个链表都无环，返回第一个相交节点，如果不想交，返回null
        :param head1:
        :param head2:
        :return:
        """
        if not head1 or not head2:
            return None

        cur1, cur2 = head1, head2
        n = 0
        while cur1.next:
            n += 1
            cur1 = cur1.next
        while cur2.next:
            n -= 1
            cur2 = cur2.next

        # 如果有交点，两个链表最后一个节点必相等
        if cur1 != cur2:
            return None

        cur1 = head1 if n > 0 else head2  # 谁长，谁的头变成cur1
        cur2 = head2 if cur1 == head1 else head1
        n = abs(n)
        while n != 0:
            cur1 = cur1.next
            n -= 1
        # 此时cur2 开始移动
        while cur1 != cur2:
            cur1 = cur1.next
            cur2 = cur2.next
        return cur1

    def bothloop(self, head1, loop1, head2, loop2):
        """
        两个有环链表，返回第一个相交节点，如果不想交返回null
        :param head1:
        :param head2:
        :return:
        """
        if loop1 == loop2:
            cur1 = head1
            cur2 = head2
            n = 0
            while cur1 != loop1:
                n += 1
                cur1 = cur1.next
            while cur2 != loop2:
                n -= 1
                cur2 = cur2.next

            cur1 = head1 if n > 0 else head2  # 谁长，谁的头变成cur1
            cur2 = head2 if cur1 == head1 else head1
            n = abs(n)
            while n != 0:
                n -= 1
                cur1 = cur1.next

            while cur1 != cur2:
                cur1 = cur1.next
                cur2 = cur2.next
            return cur1
        else:
            cur1 = loop1.next
            while cur1 != loop1:
                if cur1 == loop2:
                    return loop1  # 这里返回loop1和loop2 均可以
                cur1 = cur1.next
            return None























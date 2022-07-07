#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         RecursionAlgo
# Author:       wendi
# Date:         2022/5/25


# 递归算法
from typing import List, Tuple


class Hanoi:
    """
    汉诺塔问题
    """
    def solution1(self, n: int):
        """
        递归算法
        :param n:
        :return:
        """
        self.left2right(n)

    def left2right(self, n):
        if n == 1:
            print("Move 1 from left to right")
            return
        self.left2mid(n-1)
        print(f"Move {n} from left to right")
        self.mid2right(n-1)

    def left2mid(self, n):
        if n == 1:
            print("Move 1 from left to mid")
            return
        self.left2right(n-1)
        print(f"Move {n} from left to mid")
        self.right2mid(n-1)

    def mid2right(self, n):
        if n == 1:
            print("Move 1 from mid to right")
            return
        self.mid2left(n - 1)
        print(f"Move {n} from mid to right")
        self.left2right(n - 1)

    def mid2left(self, n):
        if n == 1:
            print("Move 1 from mid to left")
            return
        self.mid2right(n - 1)
        print(f"Move {n} from mid to left")
        self.right2left(n - 1)

    def right2left(self, n):
        if n == 1:
            print("Move 1 from right to left")
            return
        self.right2mid(n - 1)
        print(f"Move {n} from right to left")
        self.mid2left(n - 1)

    def right2mid(self, n):
        if n == 1:
            print("Move 1 from right to mid")
            return
        self.right2left(n - 1)
        print(f"Move {n} from right to mid")
        self.left2mid(n - 1)

    def solution2(self, n):
        """
        递归算法改进
        :param n:
        :return:
        """
        if n > 0:
            self.process(n, 'left', 'right', 'mid')

    def process(self, n, src, dst, other):
        if n == 1:
            print(f"Move {n} from {src} to {dst}")
        else:
            self.process(n-1, src, other, dst)
            print(f"Move {n} from {src} to {dst}")
            self.process(n-1, other, dst, src)

    class Record:
        def __init__(self, b, f, t, o):
            self.finish = False
            self.base = b
            self.src = f
            self.dst = t
            self.other = o

    def solution3(self, n):
        """
        非递归算法，结合栈实现
        :param n:
        :return:
        """
        if n < 1:
            return
        stack = []
        stack.append(Hanoi.Record(n, "left", "right", "mid"))
        while stack:
            cur = stack.pop()
            if cur.base == 1:
                print(f"Move 1 from {cur.src} to {cur.dst}")
                if stack:
                    stack[-1].finish = True
            else:
                if not cur.finish:
                    stack.append(cur)
                    stack.append(Hanoi.Record(cur.base-1, cur.src, cur.other, cur.dst))
                else:
                    print(f"Move {cur.base} from {cur.src} to {cur.dst}")
                    stack.append(Hanoi.Record(cur.base-1, cur.other, cur.dst, cur.src))


class Nqueens:
    """
    N皇后问题，回溯算法
    """
    def solution(self, n):
        if n < 1:
            return 0
        record = [0] * n
        return self.process1(0, record, n)

    def process1(self, i, record, n):
        """
        当前来到i行， 一共是0~N-1行
        在i行上放皇后，所有列都尝试
        必须要保证跟之前所有的皇后都不打架

        :param i:
        :param record: record[x] = y 之前的第x行的皇后，放在了y列上
        :param n:
        :return: 不关心i以上发生了什么，i.... 后续有多少合法的方法数
        """
        if i == n:
            return 1
        res = 0
        for j in range(0, n):
            if self.isValid(record, i, j):
                record[i] = j
                res += self.process1(i+1, record, n)
        return res

    def isValid(self, record, i, j):
        for k in range(0, i):
            # k, record[k]; i, j
            if j == record[k] or abs(k-i) == abs(record[k]-j):
                return False
        return True

    def solution1(self, n):
        if n < 1 or n > 32:
            return 0

        # 如果你是13皇后问题，limit 最右13个1，其他都是0
        limit = -1 if n == 32 else (1 << n) - 1
        return self.process2(limit, 0, 0, 0)

    def process2(self, limit, colLim, leftDiaLim, rightDiaLim):
        """

        :param limit: 如果你是13皇后问题，limit 最右13个1，其他都是0, limit 保持不变
        :param colLim: 之前皇后的列影响
        :param leftDiaLim: 之前皇后的左下对角线影响
        :param rightDiaLim: 之前皇后的右下对角线影响
        :return:
        """
        if colLim == limit:
            return 1
        # pos中所有是1的位置，是你可以去尝试皇后的位置
        pos = limit & (~(colLim | leftDiaLim | rightDiaLim))
        mostRightOne = 0
        res = 0
        while pos != 0:
            mostRightOne = pos & (~pos + 1)
            pos -= mostRightOne
            res += self.process2(limit, colLim | mostRightOne, (leftDiaLim | mostRightOne) << 1, (rightDiaLim | mostRightOne) >> 1)

        return res


class RectanglesCount:
    """
    一个平面内的n个点, 能构成多少个矩形
    """
    def solution(self, points: List[Tuple[int, int]]):
        n = len(points)

        res = 0
        for i in range(n):
            for j in range(i+1, n):
                if points[i][0] == points[j][0] or points[i][1] == points[j][1]:
                    continue
                p1 = (points[i][0], points[j][1])
                p2 = (points[i][1], points[j][0])
                if p1 in points and p2 in points:
                    res += 1
        return res/2


class JosephusProblem:
    def josephusKill(self, head, m):
        if not head or head.next == head or m < 1:
            return head

        cur = head.next
        size = 1
        while cur != head:
            size += 1
            cur = cur.next
        live = self.getLive(size, m)
        live -= 1
        while live != 0:
            live -= 1
            head = head.next
        head.next = head
        return head

    def getLive(self, i, m):
        """
        现在一共有i个节点，数到m就杀死节点，最终会活下来的节点，请返回它在有i个节点时候的编号
        :param i:
        :param m:
        :return:
        """
        if i == 1:
            return 1

        return (self.getLive(i-1, m) + m - 1) % i + 1


class RemoveInvalidParentheses:
    """
    https://leetcode.cn/problems/remove-invalid-parentheses/
    """
    def solution(self, s):
        ans = []
        self.remove(s, ans, 0, 0, ['(', ')'])
        return ans

    def remove(self, s, arr, check_index, delete_index, par):
        """
        modifyIndex <= checkIndex
        只查s[checkIndex....]的部分，因为之前的一定已经调整对了
        但是之前的部分是怎么调整对的，调整到了哪？就是modifyIndex
        比如：
        ( ( ) ( ) ) ) ...
        0 1 2 3 4 5 6
        一开始当然checkIndex = 0，modifyIndex = 0
        当查到6的时候，发现不对了，
        然后可以去掉2位置、4位置的 )，都可以
        如果去掉2位置的 ), 那么下一步就是
        ( ( ( ) ) ) ...
        0 1 2 3 4 5 6
        checkIndex = 6 ，modifyIndex = 2
        如果去掉4位置的 ), 那么下一步就是
        ( ( ) ( ) ) ...
        0 1 2 3 4 5 6
        checkIndex = 6 ，modifyIndex = 4
        也就是说，
        checkIndex和modifyIndex，分别表示查的开始 和 调的开始，之前的都不用管了  par  (  )

        :param s:
        :param arr:
        :param check_index:
        :param delete_index:
        :param par:
        :return:
        """
        cnt = 0
        for i in range(check_index, len(s)):
            # cnt 为到目前为止 ( 比 ) 多的个数
            if s[i] == par[0]:
                cnt += 1
            if s[i] == par[1]:
                cnt -= 1
            # 此时的i 为check_index 的第一个位置
            if cnt < 0:
                for j in range(delete_index, i+1):
                    if s[j] == par[1] and (j == delete_index or s[j-1] != par[1]):
                        self.remove(s[:j]+s[j+1:], arr, i, j, par)
                return

        reversed = s[::-1]
        if par[0] == '(':
            self.remove(reversed, arr, 0, 0, [')', '('])
        else:
            arr.append(reversed)


class SuperWashingMachines:
    """
    超级洗衣机问题
    https://leetcode.cn/problems/super-washing-machines/

    """
    def solution(self, arr):
        """

        :param arr:
        :return:
        """
        if not arr:
            return 0
        sum_arr = sum(arr)
        n_arr = len(arr)
        if sum_arr % n_arr != 0:
            return -1
        avg = sum_arr/n_arr
        left_sum = 0
        ans = 0
        for i in range(n_arr):
            leftRest = left_sum - i * avg
            rightRest = (sum_arr - left_sum - arr[i]) - (n_arr-i-1) * avg
            if leftRest < 0 and rightRest < 0:
                ans = max(ans, abs(leftRest) + abs(rightRest))
            else:
                ans = max(ans, max(abs(leftRest), abs(rightRest)))

            left_sum += arr[i]

        return ans


if __name__ == '__main__':
    # import datetime
    # start = datetime.datetime.now()
    # obj = Nqueens()
    # res = obj.solution(11)
    # end = datetime.datetime.now()
    # print(res)
    # print((end-start).seconds)

    s = '(()()))(()'
    obj = RemoveInvalidParentheses()
    res = obj.solution(s)
    print(res)






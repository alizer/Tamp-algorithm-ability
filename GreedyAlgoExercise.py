#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         GreedyAlgoExercise
# Author:       wendi
# Date:         2022/5/18
# Desc:         贪心算法实战
from typing import List, Set
import heapq

class BestArrange:

    class Program:
        def __init__(self, start: int, end: int):
            self.start = start
            self.end = end

    def bestArrange1(self, programs: List[Program]):
        """
        暴力方法
        :param programs:
        :return:
        """
        if not programs or len(programs) == 0:
            return 0
        return self.process(programs, 0, 0)

    def copyButExcept(self, arr: List[Program], index: int) -> List[Program]:
        copy = list()
        for i in range(len(arr)):
            if i != index:
                copy.append(arr[i])
        return copy

    def process(self, programs: List[Program], done: int, timeline: int):
        """
        目前来到timeLine的时间点，已经安排了done多的会议，剩下的会议programs可以自由安排
        :param programs: 还剩下的会议都放在programs里
        :param done: done之前已经安排了多少会议的数量
        :param timeline: timeLine目前来到的时间点是什么
        :return: 能安排的最多会议数量
        """
        if len(programs) == 0:
            return done
        # 还剩下会议
        max_value = done
        for i in range(len(programs)):
            if programs[i].start >= timeline:
                next_programs = self.copyButExcept(programs, i)
                max_value = max(max_value, self.process(next_programs, done + 1, programs[i].end))

        return max_value

    def bestArrange2(self, programs: List[Program]):
        """
        贪心法，按结束时间早的  优先安排
        :param programs:
        :return:
        """
        # 按结束时间排序
        sorted_prog = sorted(programs, key=lambda x: x.end, reverse=False)
        timeline = 0
        res = 0
        for i in sorted_prog:
            if timeline <= i.start:
                res += 1
                timeline = i.end
        return res


class Light:
    """
    https://www.nowcoder.com/questionTerminal/bb1ab2fc0f42419f986b0adc74b38398
    路灯布局，要求照亮所有地方，且使用路灯数最少
    """
    def minLight1(self, road: str):
        if not road:
            return 0
        return self.process([i for i in road], 0, set())

    def process(self, arr: List[str], index: int, lights: Set[int]):
        """

        :param arr: arr[0..index-1]已经做完决定了，那些放了灯的位置，存在lights里
        :param index: arr[index....]位置，自由选择放灯还是不放灯
        :param lights:
        :return: 要求选出能照亮所有.的方案，并且在这些有效的方案中，返回最少需要几个灯
        """
        if index == len(arr):  # arr 遍历结束
            for i in range(len(arr)):
                if arr[i] != 'X':
                    if not (i-1) in lights and not i in lights and not (i+1) in lights:
                        return float('inf')
            return len(lights)
        else:  # arr 还没结束
            # index 不放灯，直接下一个
            no = self.process(arr, index+1, lights)
            yes = float('inf')
            if arr[index] == '.':
                lights.add(index)
                yes = self.process(arr, index+1, lights)
                lights.remove(index)
            return min(no, yes)

    def minLight2(self, road: str):
        """
        贪心法
        :param road:
        :return:
        """
        arr = [i for i in road]
        i, light = 0, 0
        while i < len(arr):
            if arr[i] == 'X':
                i += 1
            else:
                light += 1
                if i + 1 == len(arr):
                    break
                else:
                    if str[i + 1] == 'X':
                        i = i + 2
                    else:
                        i = i + 3
        return light


class LessMoneySplitGold:
    """
    题目：
    一块金条切成两半，是需要花费和长度数值一样的铜板的。比如长度为20的金条，不管切成长度多大的两半，都要花费20个铜板。

    问：一群人想整分整块金条，怎么分最省铜板？
    例如，给定数组{10，20，30}，代表一共三个人，整块金条长度为10+20+30=60。
    金条要分成10，20，30。如果先把长度60的金条分成10和50，花费60；再把长度50的金条分成20和30，花费50；一共花费110铜板。
    但是如果先把长度60的金条分成30和30，花费60；再把长度30金条分成10和20，花费30；一共花费90铜板。
    输入一个数组，返回分割的最小代价。
    """
    def solution(self, arr: List[int]):
        if not arr:
            return 0
        return self.process(arr, 0)

    def process(self, arr: List[int], pre: int):
        """
        暴力法
        :param arr: 等待合并的数都在arr里，pre之前的合并行为产生了多少总代价
        :param pre:
        :return: arr中只剩一个数字的时候，停止合并，返回最小的总代价
        """
        if len(arr) == 1:
            return pre

        ans = float('inf')
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                ans = min(ans, self.process(self.copyAndMergeTwo(arr, i, j), pre+arr[i]+arr[j]))

        return ans

    def copyAndMergeTwo(self, arr: List[int], i: int, j: int):
        ans = [0] * (len(arr) - 1)
        ansi = 0
        for arri in range(len(arr)):
            if arri != i and arri != j:
                ans[ansi] = arr[arri]
                ansi += 1

        ans[ansi] = arr[i] + arr[j]
        return ans

    def solution2(self, arr: List[int]):
        """
        贪心法，结合小根堆，一次弹出最小的两个元素，然后将其之后再放进堆里，
        直到堆内元素个数为1个时，停止。
        :param arr:
        :return:
        """
        # 转换成堆结构
        heapq.heapify(arr)
        sum = 0
        while len(arr) > 1:
            item1, item2 = heapq.heappop(arr), heapq.heappop(arr)
            add = item1 + item2
            heapq.heappush(arr, add)
            sum += add
        return sum


class IPO:
    """
    输入：k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
    输出：4
    解释：
    由于你的初始资本为 0，你仅可以从 0 号项目开始。
    在完成后，你将获得 1 的利润，你的总资本将变为 1。
    此时你可以选择开始 1 号或 2 号项目。
    由于你最多可以选择两个项目，所以你需要完成 2 号项目以获得最大的资本。
    因此，输出最后最大化的资本，为 0 + 1 + 3 = 4。

    链接：https://leetcode.cn/problems/ipo
    """
    class Program_p:
        def __init__(self, p: int, c: int):
            self.p = p
            self.c = c

        def __lt__(self, other):
            if self.p >= other.p:
                return True
            else:
                return False

    class Program_c:
        def __init__(self, p: int, c: int):
            self.p = p
            self.c = c

        def __lt__(self, other):
            if self.c < other.c:
                return True
            else:
                return False

    def solution(self, k, w, profits: List[int], capital: List[int]):
        """

        :param k: 最多轮数
        :param w: 初始资本
        :param profits: 利润表
        :param capital: 成本表
        :return:
        """
        # 成本小根堆
        heaq_c = []
        # 利润大根堆
        heap_p = []
        for i in range(len(profits)):
            heapq.heappush(heaq_c, IPO.Program_c(p=profits[i], c=capital[i]))

        for i in range(k):
            while heaq_c and heaq_c[0].c <= w:
                _obj = heapq.heappop(heaq_c)
                heapq.heappush(heap_p, IPO.Program_p(_obj.p, _obj.c))

            if not heap_p:
                return w

            w += heapq.heappop(heap_p).p

        return w


class MaxPairNumber:
    """
    给定一个数组arr，代表每个人的能力值。再给定一个非负数k。
    如果两个人能力差值正好为k，那么可以凑在一起比赛，一局比赛只有两个人
    返回最多可以同时有多少场比赛
    """
    def solution(self, arr, K):
        """
        排序，优先满足较小的数，双指针同时移动
        :param arr: [1,3,5,7,8,12]
        :param K: 2
        :return: [1,3], [5,7] 最多凑齐两场比赛
        """
        if not arr or len(arr) < 2:
            return 0
        l, r = 0, 0
        n = len(arr)
        used = [False]*n

        res = 0
        while l < n and r < n:
            if used[l]:
                l += 1
            elif l == r:
                r += 1
            else:  # 不止一个数，而且都没用过！
                distance = arr[r]-arr[l]
                if distance == K:
                    res += 1
                    used[r] = True
                    l += 1
                    r += 1
                elif distance < K:
                    r += 1
                else:
                    l += 1
        return res


class BoatsToSavePeople:
    """
    给定数组 people 。people[i]表示第 i 个人的体重，船的数量不限，每艘船可以承载的最大重量为 limit。
    每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。
    返回 承载所有人所需的最小船数 。

    https://leetcode.cn/problems/boats-to-save-people/
    """
    def solution(self, arr, limit):
        """
        首尾双指针，太巧妙了！！
        :param arr:
        :param limit:
        :return:
        """
        arr.sort()
        # 单人超重，过不去
        if arr[-1] > limit:
            return -1
        l, r = 0, len(arr)-1
        res = 0
        while l <= r:
            two_sum = arr[l] if l == r else arr[l] + arr[r]
            if two_sum > limit:  # 两个人超重了，只能走一个胖子，瘦子留下来，小贪心
                r -= 1
            else:  # 两个人都能过河，同一条船，同样有个小贪心策略在里面，自己揣摩一下
                l += 1
                r -= 1
            res += 1
        return res









if __name__ == '__main__':
    obj = IPO()
    res = obj.solution(k = 3, w = 0, profits = [1,2,3], capital = [0,1,2])
    print(res)
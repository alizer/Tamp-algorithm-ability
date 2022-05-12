#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         BitOperation
# Author:       wendi
# Date:         2022/5/2

# 打印整数的二进制
import random


def printBinaryInfo(num: int):
    for i in range(31, -1, -1):
       print("0" if num & (1 << i) == 0 else "1", end='')

# 一个函数可以实现1-5的随机数，设计另一个函数，实现1-7的随机数
class Rand2rand:
    def f5(self):
        return int(random.random()*5) + 1

    # 等概率得到0和1
    def g01(self):
        while True:
            tmp = self.f5()
            if tmp < 3:
                return 0
            elif tmp > 3:
                return 1
            else:
                continue

    def f06(self):
        while True:
            tmp = (self.g01() << 2) + (self.g01() << 1) + (self.g01() << 0)
            if tmp == 7:
                continue
            else:
                return tmp

    def f17(self):
        return self.f06() + 1

    def testRes(self, testTimes: int):
        arr = [0] * 7
        for i in range(testTimes):
            tmp = self.f17()
            arr[tmp-1] += 1

        for i in range(len(arr)):
            print(f'{i+1} 出现了 {arr[i]} 次')


class Rand2randGeneral:
    """
    更一般的问题，给定一个可以产出[min, max]之间的等概率随机数函数，
    生成一个可以等概率生成[from, to]之间的随机数函数
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def randomBox(self):
        diff = int(random.random() * (self.max - self.min + 1))
        randomValue = self.min + diff
        return randomValue

    def getrand01(self):
        size = self.max - self.min + 1

        # 判断size 是奇数还是偶数
        isOdd = True if (size & 1) != 0 else False
        mid = int(size/2)

        while True:
            ans = self.randomBox() - self.min
            if isOdd and ans == mid:
                continue
            elif ans < mid:
                return 0
            else:
                return 1

    def getRandGeneral(self, fromValue, toValue):
        if fromValue == toValue:
            return fromValue

        rangeVal = toValue - fromValue
        num = 1
        while (1 << num) - 1 < rangeVal:
            num += 1

        while True:
            ans = 0
            for i in range(num):
                ans |= self.getrand01() << i
            if ans > rangeVal:
                continue
            else:
                return ans + fromValue


if __name__ == '__main__':
    # printBinaryInfo(4)
    #
    # a = 2348745
    # b = 2358787
    # print(a | b)
    # print(a & b)
    # print(a ^ b)
    # print(~a)
    # # print()
    # # 二进制位取反 再加1，得到十进制的相反数
    # print(~4 + 1)
    #
    # print(a >> 1)
    # print(random.random())

    obj = Rand2rand()
    obj.testRes(1000000)
    # n = 0
    # for i in range(1000000):
    #     n += obj.g01()
    # print(n)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         DiversitySortAlgo
# Author:       wendi
# Date:         2022/4/27
import json
import math
import random


class DiversitySort(object):
    # def __init__(self, arr):
    #     pass
    # self.arr = arr
    def generateRandomArr(self, maxLen, maxValue):
        """
        返回一个数组arr，arr长度[0,maxLen-1],arr中的每个值[0,maxValue-1]
        :return:
        """
        len = int(random.random() * maxLen)
        arr = [0] * len
        for i in range(len):
            arr[i] = int(random.random() * maxValue)
        return arr

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[i] < arr[l]:
            largest = l

        if r < n and arr[largest] < arr[r]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr):
        """
        堆排序
        时间复杂度：
        空间复杂度：
        :param arr:
        :return:
        """
        n = len(arr)

        # Build a MaxHeap
        for i in range(n, -1, -1):
            self.heapify(arr, n, i)
        print(arr)
        # 一个个交换元素
        for i in range(n - 1, 0, -1):
            # Swap item
            arr[i], arr[0] = arr[0], arr[i]
            self.heapify(arr, i, 0)

    def selectSort(self, arr):
        """
        选择最小的 往前排
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return arr

        for i in range(len(arr) - 1):
            minIndex = i
            for j in range(i, len(arr)):
                if arr[j] < arr[minIndex]:
                    minIndex = j

            arr[minIndex], arr[i] = arr[i], arr[minIndex]

        return arr

    def bubbleSort(self, arr):
        """
        两两比较，往后放
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return arr
        for i in range(len(arr) - 1, -1, -1):
            for j in range(0, i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def insertSort(self, arr):
        """
        类似于玩扑克，抓到一张新牌，然后插入到序列中
        """
        if not arr or len(arr) < 2:
            return arr
        for i in range(1, len(arr)):
            newNumIndex = i
            while newNumIndex - 1 >= 0 and arr[newNumIndex - 1] > arr[newNumIndex]:
                arr[newNumIndex], arr[newNumIndex - 1] = arr[newNumIndex - 1], arr[newNumIndex]
                newNumIndex -= 1

        return arr

    def merge(self, arr, L, M, R):
            """
            对两个有序数组归并
            :param arr:
            :param L:
            :param M:
            :param R:
            :return:
            """
            # 辅助数组
            help = [None] * (R-L+1)
            i = 0
            # 左右两个数组的起始指针位置
            p1, p2 = L, M+1
            while p1 <= M and p2 <= R:
                if arr[p1] <= arr[p2]:
                    help[i] = arr[p1]
                    p1 += 1
                else:
                    help[i] = arr[p2]
                    p2 += 1
                i += 1

            # p2 越界了
            while p1 <= M:
                help[i] = arr[p1]
                i += 1
                p1 += 1

            # p1 越界了
            while p2 <= R:
                help[i] = arr[p2]
                i += 1
                p2 += 1

            # 将help数组拷贝回主数组中
            for i in range(len(help)):
                arr[L+i] = help[i]

    def mergeSort(self, arr):
        """
        归并排序，递归方法实现
        :param arr:
        :return:
        """

        def process(arr, L, R):
            if L == R:
                return
            mid = L + int((R - L) >> 1)
            process(arr, L, mid)
            process(arr, mid+1, R)
            self.merge(arr, L, mid, R)

        if not arr or len(arr) < 2:
            return

        process(arr, 0, len(arr) - 1)

    def mergeSort2(self, arr):
        """
        归并排序，非递归方法实现
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return
        step = 1
        N = len(arr)
        while step < N:
            L = 0
            while L < N:
                # M = 0
                if N - L >= step:
                    M = L + step - 1
                else:
                    M = N - 1

                if M == N - 1:
                    break
                # R = 0
                if N - 1 - M >= step:
                    R = M + step
                else:
                    R = N - 1
                self.merge(arr, L, M, R)
                if R == N - 1:
                    break
                else:
                    L = R + 1

            if step > N/2:
                break

            step *= 2

    def netherlandsFlag(self, arr, L, R):
        """
        arr[L...R]范围上，拿arr[R]做划分值，<arr[R] =arr[R] >arr[R]
        :param arr:
        :param L:
        :param R:
        :return: 返回等于arr[R]区域的左侧和右侧位置
        """
        if L > R:
            return [-1, -1]
        if L == R:
            return [L, R]

        lessR = L - 1
        moreL = R
        index = L
        while index < moreL:
            if arr[index] < arr[R]:
                arr[index], arr[lessR+1] = arr[lessR+1], arr[index]
                index += 1
                lessR += 1
            elif arr[index] > arr[R]:
                arr[index], arr[moreL - 1] = arr[moreL - 1], arr[index]
                moreL -= 1
            else:
                index += 1

        # 此时再讲最后一个元素arr[R]和大于arr[R]区域的左侧第一个元素 交换
        arr[R], arr[moreL] = arr[moreL], arr[R]
        return [lessR+1, moreL]

    def process(self, arr, L, R):
        if L >= R:
            return
        equal_L, equal_R = self.netherlandsFlag(arr, L, R)
        self.process(arr, L, equal_L-1)
        self.process(arr, equal_R+1, R)

    def quickSort1(self, arr):
        """
        快速排序，思路类似于荷兰国旗问题
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return
        self.process(arr, 0, len(arr)-1)

    def quickSort2(self, arr):
        """
        快速排序
        改进二，每步判断是否还有大于小于区域
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return

        stack = list()
        stack.append([0, len(arr)-1])
        while stack:
            cur = stack.pop(0)
            equal_L, equal_R = self.netherlandsFlag(arr, cur[0], cur[1])
            # 有小于区域
            if equal_L > cur[0]:
                stack.append([cur[0], equal_L - 1])
            # 有大于区域
            if equal_R < cur[1]:
                stack.append([equal_R + 1, cur[1]])

    def process3(self, arr, L, R):
        if L >= R:
            return

        # 引入随机数
        randVal = int(random.random() * (R - L + 1))
        arr[randVal], arr[R] = arr[R], arr[randVal]
        equal_L, equal_R = self.netherlandsFlag(arr, L, R)
        self.process(arr, L, equal_L-1)
        self.process(arr, equal_R+1, R)

    def quickSort3(self, arr):
        """
        快速排序
        改进三，引入随机数
        :param arr:
        :return:
        """

        if not arr or len(arr) < 2:
            return
        self.process3(arr, 0, len(arr)-1)

    def CountSort(self, arr):
        """
        计数排序
        only for 0~200 value
        创建最大值+1个桶
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return
        maxValue = float("-inf")
        for item in arr:
            if item > maxValue:
                maxValue = item

        bucket = [0] * (maxValue + 1)
        for item in arr:
            bucket[item] += 1

        i = 0
        for j in range(len(bucket)):
            while bucket[j] > 0:
                bucket[j] -= 1
                arr[i] = j
                i += 1

    def RadixSort(self, arr):
        """
        基数排序
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return
        self.rsort(arr, 0, len(arr)-1, self.maxBits(arr))

    def rsort(self, arr, L, R, digit):
        radix = 10
        help = [None] * (R - L + 1)
        for d in range(1, digit+1):
            count = [0] * radix
            for i in range(L, R+1):
                j = self.getDigit(arr[i], d)
                count[j] += 1
            # 累加和
            for i in range(1, radix):
                count[i] = count[i] + count[i-1]

            for i in range(R, L-1, -1):
                j = self.getDigit(arr[i], d)
                help[count[j] - 1] = arr[i]
                count[j] -= 1

            j = 0
            for i in range(L, R+1):
                arr[i] = help[j]
                j += 1

    def maxBits(self, arr):
        """
        返回最大值的位数
        :param arr:
        :return:
        """
        maxValue = float("-inf")
        for item in arr:
            if item > maxValue:
                maxValue = item
        res = 0
        while maxValue != 0:
            maxValue = maxValue // 10
            res += 1
        return res

    def getDigit(self, x, d):
        """
        返回x这个数的第d位的数据
        :param x:
        :param d:
        :return:
        """
        return int((x // math.pow(10, d-1)) % 10)


if __name__ == '__main__':
    # arr = [12, 11, 13, 5, 6, 7]
    obj = DiversitySort()
    # obj.insertSort(arr)
    arr = obj.generateRandomArr(9, 100)
    print(arr)
    obj.RadixSort(arr)
    print(arr)

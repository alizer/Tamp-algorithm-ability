#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         HeapExercise
# Author:       wendi
# Date:         2022/5/8

from typing import TypeVar

class MaxHeap:
    """
    手写大根堆
    """
    def __init__(self, limit):
        """
        :param heap: 大根堆，数组形式
        :param limit: 大根堆最大size
        :param heapSize: 当前堆size
        """
        self.heap = [None] * limit
        self.limit = limit
        self.heapSize = 0

    def isEmpty(self):
        return len(self.heap) == 0

    def isFull(self):
        return len(self.heap) == self.limit

    def push(self, value):
        if self.heapSize == self.limit:
            raise Exception("heap is full!")

        self.heap[self.heapSize] = value
        self.heapInsert(self.heap, self.heapSize)
        self.heapSize += 1

    # 新加进来的数，放在了index位置，然后依次向上和父节点比较，
    # 移动到0位置或比不过自己的父节点了，停止移动。
    def heapInsert(self, arr, index):
        """
        从index位置往上看
        :param arr:
        :param index:
        :return:
        """
        # 父节点索引 int((index-1)/2)
        while arr[index] > arr[int((index-1)/2)]:
            arr[index], arr[int((index-1)/2)] = arr[int((index-1)/2)], arr[index]
            index = int((index-1)/2)

    def pop(self):
        """
        移除堆顶值，且移除堆顶值后剩下的数，仍然满足大根堆或小根堆
        :return: 堆顶值
        """
        res = self.heap[0]
        self.heapSize -= 1
        self.heap[0], self.heap[self.heapSize] = self.heap[self.heapSize], self.heap[0]
        self.heapify(self.heap, 0, self.heapSize)
        return res

    def heapify(self, heap, index, heapSize):
        """
        从index位置往下看，不断的下沉
        停止条件：
        1、较大的孩子节点都比不过父节点了
        2、没有孩子节点了
        """
        # 如果有左孩子，右孩子可能有，也可能没有
        left = index * 2 + 1
        while left < heapSize:
            # 取较大孩子的下标
            if left + 1 < heapSize and heap[left+1] > heap[left]:
                largest = left + 1
            else:
                largest = left

            # 再和父节点比较
            if heap[largest] < heap[index]:
                largest = index

            if largest == index:
                break

            heap[largest], heap[index] = heap[index], heap[largest]
            index = largest
            left = 2 * index + 1


# class HeapGreater:
#     def __init__(self):
#         T = TypeVar("T")


if __name__ == '__main__':

    mHeap = MaxHeap(limit=7)
    mHeap.push(8)
    mHeap.push(6)
    mHeap.push(3)
    mHeap.push(5)
    mHeap.push(4)
    mHeap.push(7)
    mHeap.push(2)
    for i in range(7):
        print(mHeap.pop())
        print(mHeap.heap)
















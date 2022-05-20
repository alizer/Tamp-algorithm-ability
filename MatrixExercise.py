#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         MatrixExercise
# Author:       wendi
# Date:         2022/5/19

# 矩阵处理技巧
from typing import List


class ZigZagPrintMatrix:
    """
    Z字形 打印矩阵
    """
    def solution(self, matrix: List[List[int]]) -> None:
        """
        tR，dR 分别是当前阶段的上下行号
        tC，dC 分别是当前阶段的左右列号
        :param matrix:
        :return:
        """
        tR, dR = 0, 0
        tC, dC = 0, 0
        endR = len(matrix) - 1
        endC = len(matrix[0]) - 1
        fromUp = False
        while tR != endR + 1:
            self.printLevel(matrix, tR, tC, dR, dC, fromUp)
            tR = tR + 1 if tC == endC else tR
            tC = tC if tC == endC else tC + 1
            dC = dC + 1 if dR == endR else dC
            dR = dR if dR == endR else dR + 1
            fromUp = not fromUp

    def printLevel(self, matrix: List[List[int]], tR: int, tC: int, dR: int, dC: int, direction: bool):
        if direction:
            while tR != dR + 1:
                print(matrix[tR][tC], end=' ')
                tR += 1
                tC -= 1
        else:
            while dR != tR - 1:
                print(matrix[dR][dC], end=' ')
                dR -= 1
                dC += 1


class RotateMatrix:
    """
    旋转矩阵，类似于玩魔方
    """
    def solution(self, matrix: List[List[int]]):
        a, b = 0, 0
        c, d = len(matrix) - 1, len(matrix[0]) - 1
        while a < c:
            self.rotateEdge(matrix, a, b, c, d)
            a += 1
            b += 1
            c -= 1
            d -= 1
        return matrix

    def rotateEdge(self, m, a, b, c, d):
        for i in range(d-b):
            tmp = m[a][b+i]
            m[a][b + i] = m[c - i][b]
            m[c - i][b] = m[c][d - i]
            m[c][d - i] = m[a + i][d]
            m[a + i][d] = tmp


class PrintMatrixSpiralOrder:
    """
    螺旋顺序  打印矩阵
    """
    def solution(self, matrix: List[List[int]]):
        tr, tc = 0, 0
        dr, dc = len(matrix) - 1, len(matrix[0]) - 1
        while tr <= dr and tc <= dc:
            self.printEdge(matrix, tr, tc, dr, dc)
            tr += 1
            tc += 1
            dr -= 1
            dc -= 1

    def printEdge(self, matrix, tr, tc, dr, dc):
        if tr == dr:
            for i in range(tc, dc+1):
                print(matrix[tr][i], end=' ')
        elif tc == dc:
            for i in range(tr, dr+1):
                print(matrix[i][tc], end=' ')
        else:
            cur_c = tc
            cur_r = tr
            while cur_c != dc:
                print(matrix[tr][cur_c], end=' ')
                cur_c += 1
            while cur_r != dr:
                print(matrix[cur_r][dc], end=' ')
                cur_r += 1
            while cur_c != tc:
                print(matrix[dr][cur_c], end=' ')
                cur_c -= 1
            while cur_r != tr:
                print(matrix[cur_r][tc], end=' ')
                cur_r -= 1


if __name__ == '__main__':
    obj = PrintMatrixSpiralOrder()
    res = obj.solution(matrix=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # print(res)
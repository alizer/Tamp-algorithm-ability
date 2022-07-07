#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         GraphExercise
# Author:       wendi
# Date:         2022/5/20

#  图算法练习
import heapq
from typing import Dict, Set, List


class Node:
    """
    节点结构
    """
    def __init__(self, value: int):
        self.value = value
        self.in_degree = 0
        self.out_degree = 0
        self.next_nodes = []
        self.edges = []


class Edge:
    """
    边结构
    """
    def __init__(self, weight: int, src: Node, dst: Node):
        self.weight = weight
        self.src = src
        self.dst = dst

    def __lt__(self, other):
        """
        相当于比较器，为了排序用
        :param other:
        :return:
        """
        if self.weight < other.weight:
            return True
        else:
            return False


class Graph:
    """
    图结构
    """
    def __init__(self):
        self.nodes = dict()
        self.edges = set()


class GraphGenerator:
    """
    创建图
    """
    def createGraph(self, arr: List[List[int]]):
        graph = Graph()
        for i in range(len(arr)):
            weight = arr[i][0]
            src = arr[i][1]
            dst = arr[i][2]
            if src not in graph.nodes.keys():
                graph.nodes[src] = Node(src)

            if dst not in graph.nodes.keys():
                graph.nodes[dst] = Node(dst)

            src_node = graph.nodes.get(src)
            dst_node = graph.nodes.get(dst)
            edge = Edge(weight, src_node, dst_node)
            graph.edges.add(edge)

            src_node.out_degree += 1
            dst_node.in_degree += 1
            src_node.next_nodes.append(dst_node)
            src_node.edges.append(edge)

        return graph


class BFS:
    """
    图的宽度优先遍历
    """
    def bfs(self, start: Node):
        """
        从start这个node出发，进行宽度优先遍历
        :param start:
        :return:
        """
        if not start:
            return
        queue = list()
        node_set = set()
        queue.append(start)
        node_set.add(start)
        while queue:
            cur = queue.pop(0)
            print(cur.value)
            for node in cur.next_nodes:
                if node not in node_set:
                    node_set.add(node)
                    queue.append(node)


class DFS:
    """
    图的深度优先遍历
    """
    def dfs(self, start: Node):
        """
        从start这个node出发，进行深度优先遍历
        :param start:
        :return:
        """
        if not start:
            return
        stack = []
        node_set = set()
        stack.append(start)
        node_set.add(start)
        print(start.value)

        while stack:
            cur = stack.pop()
            for node in cur.next_nodes:
                if node not in node_set:
                    stack.append(cur)
                    stack.append(node)
                    node_set.add(node)
                    print(node.value)
                    break


class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = list()


class Record:
    def __init__(self, n: DirectedGraphNode, o: int):
        self.node = n
        self.deep = o


class TopologicalOrder:
    """
    图的拓扑排序
    即递归移除入度为0的节点
    """

    def dfs(self, graph: List[DirectedGraphNode]):

        def f(cur: DirectedGraphNode, order: Dict[DirectedGraphNode, Record]):
            if cur in order.keys():
                return order.get(cur)
            follow = 0
            for node in cur.neighbors:
                follow = max(follow, f(node, order).deep)
            ans = Record(cur, follow+1)
            order[cur] = ans

        order = dict()
        for cur in graph:
            f(cur, order)
        recordArr = list()
        for r in order.values():
            recordArr.append(r)
        recordArr.sort(key=lambda x: x.deep, reverse=False)
        ans = list()
        for r in recordArr:
            ans.append(r)
        return ans


    def bfs(self, graph: List[DirectedGraphNode]):
        """
        等同于sort方法，只是图结构不同
        :param graph:
        :return:
        """
        in_degree_map = dict()
        for cur in graph:
            in_degree_map[cur] = 0

        for cur in graph:
            for nei_node in cur.neighbors:
                in_degree_map[nei_node] = in_degree_map.get(nei_node) + 1

        # 只有剩余入度为0的点，才进入这个队列
        zero_queue = list()
        for cur in in_degree_map.keys():
            if in_degree_map[cur] == 0:
                zero_queue.append(cur)

        ans = list()
        while zero_queue:
            cur = zero_queue.pop(0)
            ans.append(cur)
            for node in cur.neighbors:
                in_degree_map[node] = in_degree_map.get(node) - 1
                if in_degree_map[node] == 0:
                    zero_queue.append(node)

        return ans

    def sort(self, graph: Graph):
        # key 某个节点   value 剩余的入度
        in_degree_dc = dict()
        # 只有剩余入度为0的点，才进入这个队列
        zero_queue = list()
        for node in graph.nodes.values():
            in_degree_dc[node] = node.in_degree
            if node.in_degree == 0:
                zero_queue.append(node)

        res = list()
        while zero_queue:
            cur = zero_queue.pop(0)
            res.append(cur)

            for node in cur.next_nodes:
                in_degree_dc[node] = in_degree_dc.get(node) - 1
                if in_degree_dc.get(node) == 0:
                    zero_queue.append(node)

        return res


class UnionFind:
    """
    并查集基础算法
    """
    def __init__(self):
        self.parents = dict()
        self.size_map = dict()

    def makeSets(self, nodes: List[Node]):
        self.parents.clear()
        self.size_map.clear()

        for node in nodes:
            self.parents[node] = node
            self.size_map[node] = 1

    def findFather(self, cur: Node):
        """
        给你一个节点，请你往上到不能再往上，把代表节点返回
        :param cur:
        :return:
        """
        path_stack = []
        while cur != self.parents.get(cur):
            path_stack.append(cur)
            cur = self.parents.get(cur)

        while path_stack:
            self.parents[path_stack.pop()] = cur

        return cur

    def isSameSet(self, a, b):
        return self.findFather(a) == self.findFather(b)

    def union(self, a, b):
        a_head = self.findFather(a)
        b_head = self.findFather(b)

        if a_head != b_head:
            a_set_size = self.size_map.get(a_head)
            b_set_size = self.size_map.get(b_head)
            big = a_head if a_set_size >= b_set_size else b_head
            small = b_head if big == a_head else a_head
            self.parents[small] = big
            self.size_map[big] = a_set_size + b_set_size
            self.size_map.pop(small)

        return self.size_map.__len__()

    def getSetNum(self):
        return len(self.size_map)


class Kruskal:
    """
    最小生成树算法：常用的两个算法 K算法和P算法！！

    K算法：克鲁斯卡尔算法，将所有点连接起来的总权重最小的边集合。【邮递员问题】
    总是从权值最小的边开始考虑，依次考察权值依次变大的边
    当前的边要么进入最小生成树的集合，要么丢弃
    如果当前的边进入最小生成树的集合中，不会形成环，就要当前边，否则不要
    考察完所有边之后，最小生成树的集合就得到了
    """
    def kruskalMST(self, graph: Graph):
        unionfind = UnionFind()
        unionfind.makeSets(graph.nodes.values())

        # 从小的边到大的边，依次弹出，小根堆！
        hp = []
        for edge in graph.edges:
            heapq.heappush(hp, edge)

        res = set()
        while hp:
            edge = heapq.heappop(hp)
            if not unionfind.isSameSet(edge.src, edge.dst):
                res.add(edge)
                unionfind.union(edge.src, edge.dst)

        return res


class Prim:
    """
    输入：指定任意一个出发点
    创建一个点集合，保留所有被解锁过的点
    每个边有四种状态：锁定、被解锁、选中、丢弃
    步骤：点->所有相邻的边，左右两端有不在点集之中的点的边中取最小权重->点->repeat
    """
    def primMst(self, graph: Graph):
        # 解锁的边进入小根堆
        heap_edge = list()
        # 哪些点被解锁出来了
        nodeSet = set()
        # 被选中的边放在结果集合中
        result = set()

        for node in graph.nodes.values():  # 遍历每个点是避免有多个连通子图
            if node not in nodeSet:
                nodeSet.add(node)
                for edge in node.edges:
                    heapq.heappush(heap_edge, edge)

                while heap_edge:
                    edge = heapq.heappop(heap_edge)
                    dst = edge.dst
                    if dst not in nodeSet:
                        nodeSet.add(dst)
                        for nextEdge in dst.edges:
                            heapq.heappush(heap_edge, nextEdge)


class Dijkstra:
    """
    单源最短路径，边权重非负
    """
    def dijkstra1(self, src: Node):
        """
        :param src:
        :return: 从src出发 到所有点的最短距离
        """
        # key：从src出发到达的节点；value:从src出发到达的节点的最短距离，
        # 若没有节点的记录，认为是正无穷
        distanceMap = dict()
        distanceMap[src] = 0
        selectedNodes = set()
        minNode = self.getMinDistanceAndUnselectedNode(distanceMap, selectedNodes)
        while minNode:
            distance = distanceMap.get(minNode)
            for edge in minNode.edges:
                dstNode = edge.dst
                if dstNode not in distanceMap.keys():
                    distanceMap[dstNode] = distance + edge.weight
                else:
                    distanceMap[edge.dst] = min(distanceMap.get(dstNode), distance + edge.weight)

            selectedNodes.add(minNode)
            minNode = self.getMinDistanceAndUnselectedNode(distanceMap, selectedNodes)

        return distanceMap

    def getMinDistanceAndUnselectedNode(self, distanceMap, touchNodes):
        minNode = None
        minDistance = float('inf')
        for node, distance in distanceMap.items():
            if node not in touchNodes and distance < minDistance:
                minNode = node
                minDistance = distance
        return minNode

    class NodeRecord:
        def __init__(self, node: Node, distance: int):
            self.node = node
            self.distance = distance

    class NodeHeap:
        def __init__(self, size: int):
            # 实际的堆结构
            self.nodes = [] * size
            # key 某一个node， value 上面堆中的位置
            self.heapIndexMap = dict()
            # key 某一个节点， value 从源节点出发到该节点的目前最小距离
            self.distanceMap = dict()
            # 堆上有多少个点
            self.size = 0

        def isEmpty(self):
            return self.size == 0

        def addOrUpdateOrIgnore(self, node: Node, distance: int):
            """
            有一个点叫node，现在发现了一个从源节点出发到达node的距离为distance
            判断要不要更新，如果需要的话，就更新
            :param node:
            :param distance:
            :return:
            """
            if self.inHeap(node):
                self.distanceMap[node] = min(self.distanceMap.get(node), distance)
                self.insertHeapify(node, self.heapIndexMap.get(node))
            if not self.isEntered(node):
                self.nodes[self.size] = node
                self.heapIndexMap[node] = self.size
                self.distanceMap[node] = distance
                self.insertHeapify(node, self.size)
                self.size += 1

        def isEntered(self, node: Node):
            return self.heapIndexMap.__contains__(node)

        def inHeap(self, node: Node):
            return self.isEntered(node) and self.heapIndexMap.get(node) != -1

        def insertHeapify(self, node: Node, index: int):
            while self.distanceMap.get(self.nodes[index]) < self.distanceMap.get(self.nodes[(index-1)//2]):
                self.swap(index, (index-1)//2)
                index = (index-1)//2

        def heapify(self, index: int, size: int):
            left = index * 2 + 1
            while left < size:
                if left + 1 < size and self.distanceMap.get(self.nodes[left+1]) < self.distanceMap.get(self.nodes[left]):
                    smallest = left+1
                else:
                    smallest = left

                smallest = smallest if self.distanceMap.get(self.nodes[smallest]) < self.distanceMap.get(self.nodes[index]) else index

                self.swap(smallest, index)
                index = smallest
                left = index * 2 + 1

        def swap(self, index1: int, index2: int):
            self.heapIndexMap[self.nodes[index1]] = index2
            self.heapIndexMap[self.nodes[index2]] = index1
            self.nodes[index1], self.nodes[index2] = self.nodes[index2],  self.nodes[index1]

        def pop(self):
            nodeRecord = Dijkstra.NodeRecord(self.nodes[0], self.distanceMap.get(self.nodes[0]))
            self.swap(0, self.size - 1)
            self.heapIndexMap[self.nodes[self.size-1]] = -1
            self.distanceMap.pop(self.nodes[self.size-1])
            self.nodes[self.size-1] = None
            self.heapify(0, self.size-1)
            self.size -= 1
            return nodeRecord

    def dijkstra2(self, src: Node):
        """
        改写堆
        :param src:
        :return:
        """
        nodeHeap = Dijkstra.NodeHeap()
        nodeHeap.addOrUpdateOrIgnore(src, 0)
        result = dict()
        while nodeHeap:
            record = nodeHeap.pop()
            cur = record.node
            distance = record.distance
            for edge in cur.edges:
                nodeHeap.addOrUpdateOrIgnore(edge.dst, edge.weight+distance)
            result[cur] = distance

        return result


if __name__ == '__main__':
    obj = GraphGenerator()
    graph = obj.createGraph(arr=[[1, 0, 1], [3, 0, 4], [100, 1, 4], [2, 1, 2], [4, 2, 3], [50, 3, 4]])
    krs = Kruskal()
    res = krs.kruskalMST(graph)
    print([edge.weight for edge in res])

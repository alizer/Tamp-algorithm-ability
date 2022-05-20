#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         TreeExercise
# Author:       wendi
# Date:         2021/10/30
from typing import List


class BinaryTree(object):
    """
    二叉树
    """
    def __init__(self, val):
        super(BinaryTree, self).__init__()
        self.value = val
        self.left = None
        self.right = None


class naryTree(object):
    """
    多叉树
    """
    def __init__(self, val, child):
        super(naryTree, self).__init__()
        self.value = val
        self.children = child


class TrieTree(object):
    """
    字典树【前缀数】
    """
    def __init__(self):
        super(TrieTree, self).__init__()
        self.children = [None] * 26
        self.isEnd = False

    def searchPrefix(self, prefix: str) -> 'TrieTree':
        node = self
        for ch in prefix:
            idx = ord(ch) - ord('a')
            if not node.children[idx]:
                return None
            node = node.children[idx]
        return node

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            idx = ord(ch) - ord('a')
            if not node.children[idx]:
                node.children[idx] = TrieTree()
            node = node.children[idx]
        node.isEnd = True

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        return self.searchPrefix(prefix) is not None


class TrieNode(object):
    def __init__(self):
        super(TrieNode, self).__init__()
        self.child = {}
        self.flag = None


class LcpTree(object):
    def __init__(self):
        super(LcpTree, self).__init__()
        self.root = TrieNode()

    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        elif len(strs) == 1:
            return strs[0]

        # 将strs中所有字符串插入Trie树
        for words in strs:
            curNode = self.root
            for word in words:
                if curNode.child.get(word) is None:
                    curNode.child[word] = TrieNode()
                curNode = curNode.child[word]
            curNode.flag = 1

        curNode = self.root

        lcp = []
        while curNode.flag != 1:
            # 遍历Trie树，直至当前节点的子节点分叉数大于1
            if len(curNode.child) == 1:
                curNodeChar = list(curNode.child.keys())[0]
                lcp.append(curNodeChar)
                curNode = curNode.child[curNodeChar]
            else:
                break
        return ''.join(lcp)


class RecursiveTraversalBT:
    """
    递归遍历二叉树
    """
    def preOrder(self, head: BinaryTree):
        """
        先序遍历
        :param head:
        :return:
        """
        if not head:
            return

        print(head.value)
        self.preOrder(head.left)
        self.preOrder(head.right)

    def inOrder(self, head: BinaryTree):
        """
        中序遍历
        :param head:
        :return:
        """
        if not head:
            return

        self.inOrder(head.left)
        print(head.value)
        self.inOrder(head.right)

    def postOrder(self, head: BinaryTree):
        """
        后序遍历
        :param head:
        :return:
        """
        if not head:
            return

        self.postOrder(head.left)
        self.postOrder(head.right)
        print(head.value)


class UnRecursiveTraversalBT:
    """
    非递归遍历二叉树
    """
    def preOrder(self, head: BinaryTree):
        """
        先序遍历, 结合栈
        :param head:
        :return:
        """
        if head:
            stack = [head]
            while stack:
                head = stack.pop()
                print(head.value)
                if head.right:
                    stack.append(head.right)
                if head.left:
                    stack.append(head.left)

    def inOrder(self, head: BinaryTree):
        """
        中序遍历
        :param head:
        :return:
        """
        if head:
            stack = []
            while stack or head:
                if head:
                    stack.append(head)
                    head = head.left
                else:
                    head = stack.pop()
                    print(head.value)
                    head = head.right

    def postOrder(self, head: BinaryTree):
        """
        先序遍历, 使用两个栈
        :param head:
        :return:
        """
        if head:
            stack1 = []
            stack2 = []
            stack1.append(head)
            while stack1:
                head = stack1.pop()
                stack2.append(head)
                if head.left:
                    stack2.append(head.right)
                if head.right:
                    stack2.append(head.left)

            while stack2:
                print(stack2.pop().value)

    def postOrder2(self, head: BinaryTree):
        """
        先序遍历, 使用1个栈
        :param head:
        :return:
        """
        if head:
            stack = [head]
            while stack:
                c = stack[-1]
                if c.left and head != c.left and head != c.right:
                    stack.append(c.left)
                elif c.right and head != c.right:
                    stack.append(c.right)
                else:
                    print(stack.pop().value)
                    head = c


class LevelTraversalBT:
    """
    二叉树 按层遍历
    """
    def solution(self, head: BinaryTree):
        if not head:
            return
        queue = list()
        queue.append(head)
        while queue:
            cur = queue.pop(0)
            print(cur.value)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)


class SerializeAndReconstructTree:
    """
    /*
     * 二叉树可以通过先序、后序或者按层遍历的方式序列化和反序列化，
     * 以下代码全部实现了。
     * 但是，二叉树无法通过中序遍历的方式实现序列化和反序列化
     * 因为不同的两棵树，可能得到同样的中序序列，即便补了空位置也可能一样。
     * 比如如下两棵树
     *         __2
     *        /
     *       1
     *       和
     *       1__
     *          \
     *           2
     * 补足空位置的中序遍历结果都是{ null, 1, null, 2, null}
     *
     * */
    """
    def preSerial(self, head: BinaryTree):
        ans = []

        def pres(node, arr):
            if not node:
                arr.append(None)
            else:
                arr.append(node.value)
                pres(node.left, arr)
                pres(node.right, arr)

        pres(head, ans)
        return ans

    def posSerial(self, head: BinaryTree):
        ans = []

        def poss(node, arr):
            if not node:
                arr.append(None)
            else:
                poss(node.left, arr)
                poss(node.right, arr)
                arr.append(node.value)

        poss(head, ans)
        return ans

    def levelSerial(self, head: BinaryTree):
        """
        层序遍历 序列化
        :param head:
        :return:
        """
        ans = []
        if not head:
            ans.append(None)
        else:
            ans.append(head.value)
            queue = [head]

            while queue:
                head = queue.pop(0)
                if head.left:
                    ans.append(head.left.value)
                    queue.append(head.left)
                else:
                    ans.append(None)

                if head.right:
                    ans.append(head.right.value)
                    queue.append(head.right)
                else:
                    ans.append(None)
        return ans

    def buildByPreQueue(self, prelist):
        def preb(prelist):
            value = prelist.pop(0)
            if not value:
                return None
            head = BinaryTree(value)
            head.left = preb(prelist)
            head.right = preb(prelist)
            return head

        if not prelist or len(prelist) == 0:
            return None
        return preb(prelist)

    def buildByPosQueue(self, poslist):
        def posb(posstack):
            value = posstack.pop()
            if not value:
                return None
            head = BinaryTree(value)
            head.right = posb(posstack)
            head.left = posb(posstack)
            return head

        if not poslist or len(poslist) == 0:
            return None

        #  左右中  ->  stack(中右左)
        stack = []
        while poslist:
            stack.append(poslist.pop(0))
        return posb(poslist)

    def buildByLevelQueue(self, levelList):
        """
        根据层序队列 反序列化
        :param poslist:
        :return:
        """
        def generateNode(val):
            if not val:
                return None
            return BinaryTree(val)
        if not levelList or len(levelList) == 0:
            return None

        head = generateNode(levelList.pop(0))
        queue = []
        if head:
            queue = [head]

        while queue:
           node = queue.pop(0)
           node.left = generateNode(levelList.pop(0))
           node.right = generateNode(levelList.pop(0))
           if node.left:
               queue.append(node.left)

           if node.right:
               queue.append(node.right)
        return head


class EncodeNaryTreeToBinaryTree:
    """
    多叉树转二叉树
    转换规则：将一棵多叉树转换成二叉树，我们遵循的原则是：左儿子，右兄弟。
    算法描述：将多叉树的第一个儿子结点作为二叉树的左结点，将其兄弟结点作为二叉树的右结点。

    (1)T中的结点与K中的结点一一对应。
    (2)T中的某个结点N的第一个子结点为N1，则K中N1为N的左儿子结点
    (3)T中的某个结点N的第i个子结点记为Ni(除第一个子结点)，则K中Ni为Ni-1的右儿子结点(N2为N1的右儿子结点，N3为N2的右儿子结点)


    """
    def encode(self, root: naryTree):
        """
        Encodes an n-ary tree to a binary tree.
        :param root:
        :return:
        """
        if not root:
            return None
        head = BinaryTree(root.value)
        head.left = self.en(root.children)
        return head

    def en(self, children: List[naryTree]):
        head, cur = None, None
        for child in children:
            tNode = BinaryTree(child.value)
            if not head:
                head = tNode
            else:
                cur.right = tNode

            cur = tNode
            cur.left = self.en(child.children)

        return head

    def decode(self, root: BinaryTree):
        """
        Decodes your binary tree to an n-ary tree.
        :param root:
        :return:
        """
        if not root:
            return None
        return naryTree(root.value, self.de(root.left))

    def de(self, root: BinaryTree):
        children = []
        while root:
            cur = naryTree(root.value, self.de(root.left))
            children.append(cur)
            root = root.right

        return children


class PrintBinaryTre:
    """
    打印二叉树
    """
    def printTree(self, head: BinaryTree):
        print("Binary Tree:")
        self.printInorder(head, 0, "H", 17)

    def printInorder(self, head, height, to, len):
        if not head:
            return
        self.printInorder(head.right, height+1, "v", len)
        val = to + str(head.value) + to
        lenM = len(val)
        lenL = (len - lenM) / 2
        lenR = len - lenM - lenL
        val = self.getSpace(lenL) + val + self.getSpace(lenR)
        print(self.getSpace(height*len + val))
        self.printInorder(head.left, height+1, "^", len)

    def getSpace(self, num):
        space = " "
        buf = ""
        for i in range(num):
            buf += space
        return buf


class MaxWidthLevel:
    """
    二叉树最大宽度
    """
    def solution(self, head: BinaryTree):
        if not head:
            return 0
        queue = []
        queue.append(head)

        curEnd = head  # 当前层，最右节点是谁
        nextEnd = None  # 下一层，最右节点是谁
        curLevelNodes = 0  # 当前层的节点数
        maxWidth = 0
        while queue:
            cur = queue.pop(0)
            if cur.left:
                queue.append(cur.left)
                nextEnd = cur.left
            if cur.right:
                queue.append(cur.right)
                nextEnd = cur.right

            curLevelNodes += 1

            if cur == curEnd:
                maxWidth = max(maxWidth, curLevelNodes)
                curLevelNodes = 0
                curEnd = nextEnd
        return maxWidth


class IsCBT:
    """
    是否 完全二叉树
    """
    def solution(self, head: BinaryTree):
        """
        使用队列
        :param head:
        :return:
        """
        if not head:
            return True
        queue = list()

        # 是否遇到过左右两个孩子不双全的节点
        leaf = False
        l, r = None, None
        queue.append(head)
        while queue:
            head = queue.pop(0)
            l, r = head.left, head.right
            if (leaf and (l or r)) or (not l and r):  # 如果遇到了不双全的节点之后，又发现当前节点不是叶节点
                return False

            if l:
                queue.append(l)

            if r:
                queue.append(r)

            if not l or not r:
                leaf = True

        return True

    class BtInfo:
        def __init__(self, full, cbt, h):
            """
            对每一棵子树，是否是满二叉树、是否是完全二叉树、高度
            :param full:
            :param cbt:
            :param h:
            """
            self.isFull = full
            self.isCBT = cbt
            self.height = h

    def solution2(self, head: BinaryTree):
        if not head:
            return True
        return self.process(head).isCBT

    def process(self, node: BinaryTree):
        if not node:
            return IsCBT.BtInfo(True, True, 0)

        leftInfo = self.process(node.left)
        rightInfo = self.process(node.right)

        height = max(leftInfo.height, rightInfo.height) + 1
        isFull = leftInfo.isFull and rightInfo.isFull and leftInfo.height == rightInfo.height

        isCBT = False
        if isFull:
            isCBT = True
        else:  # 以x为头整棵树，不满
            if leftInfo.isCBT and rightInfo.isCBT:
                if leftInfo.isCBT and rightInfo.isFull and leftInfo.height == rightInfo.height + 1:
                    isCBT = True

                if leftInfo.isFull and rightInfo.isFull and leftInfo.height == rightInfo.height + 1:
                    isCBT = True

                if leftInfo.isCBT and rightInfo.height == rightInfo.height:
                    isCBT = True

        return IsCBT.BtInfo(isFull, isCBT, height)


class IsBST:
    """
    是否二叉搜索树
    """
    def inOrder(self, head: BinaryTree, arr: List[int]):
        if not head:
            return
        self.inOrder(head.left)
        arr.append(head.value)
        self.inOrder(head.right)

    def solution(self, head: BinaryTree):
        """
        中序遍历，然后逐对判断
        :param head: 
        :return: 
        """
        if not head:
            return True
        arr = []
        self.inOrder(head, arr)
        for i in range(1, len(arr)):
            if arr[i].value < arr[i-1].value:
                return False

        return True
    
    class BstInfo:
        def __init__(self, i: bool, ma: int, mi: int):
            self.isBST = i
            self.max = ma
            self.min = mi
            
    def solution2(self, head: BinaryTree):
        """
        递归套路
        :param head: 
        :return: 
        """
        if not head:
            return True
        return self.process(head).isBST
    
    def process(self, head: BinaryTree):
        if not head:
            return None
        
        leftInfo = self.process(head.left)
        rightInfo = self.process(head.right)
        
        max_val = head.value
        min_val = head.value
        if leftInfo:
            max_val = max(max_val, leftInfo.max)
        if rightInfo:
            max_val = max(max_val, rightInfo.max)
        
        if leftInfo:
            min_val = min(min_val, leftInfo.min)

        if rightInfo:
            min_val = min(min_val, rightInfo.min)

        isBST = True
        if leftInfo and not leftInfo.isBST:
            return False
        if rightInfo and not rightInfo.isBST:
            return False

        if leftInfo and leftInfo.max >= head.value:
            return False
        if rightInfo and rightInfo.min <= head.value:
            return False

        return IsBST.BstInfo(isBST, max_val, min_val)
        

class IsBalanced:
    """
    是否平衡二叉树
    """
    def solution1(self, head: BinaryTree):
        """
        迭代法
        :param head:
        :return:
        """
        if not head:
            return True
        res = [True]
        self.process(head, res)
        return res

    def process(self, head: BinaryTree, res: List[bool]):
        if not res[0] or not head:
            return -1
        leftHeight = self.process(head.left, res)
        rightHeight = self.process(head.right, res)
        if abs(leftHeight - rightHeight) > 1:
            res[0] = False

        return max(leftHeight, rightHeight) + 1

    class IsBalaInfo:
        def __init__(self, i: bool, h: int):
            self.isBalanced = i
            self.height = h

    def solution2(self, head: BinaryTree):
        """
        递归套路
        :param head:
        :return:
        """
        if head is None:
            return IsBalanced.IsBalaInfo(True, 0)

        leftInfo = self.solution2(head.left)
        rightInfo = self.solution2(head.right)
        height = max(leftInfo.height, rightInfo.height) + 1
        isBalanced = True
        if not leftInfo.isBalanced:
            isBalanced = False
        if not rightInfo.isBalanced:
            isBalanced = False
        if abs(leftInfo.height - rightInfo.height) > 1:
            isBalanced = False

        return IsBalanced.IsBalaInfo(isBalanced, height)


class IsFull:
    """
    是否满二叉树
    """
    class IsFullInfo:
        def __init__(self, h: int, n: int):
            self.height = h
            self.nodes = n

    def solution1(self, head: BinaryTree):
        """
        递归套路
        :param head:
        :return:
        """
        if head is None:
            return True
        info = self.process(head)
        return (1 << info.height) - 1 == info.nodes

    def process(self, head: BinaryTree):
        if head is None:
            return IsFull.IsFullInfo(0, 0)

        leftInfo = self.process(head.left)
        rightInfo = self.process(head.right)
        height = max(leftInfo.height, rightInfo.height) + 1
        nodes = leftInfo.nodes + rightInfo.nodes + 1
        return IsFull.IsFullInfo(height, nodes)

    class IsFullInfo2:
        def __init__(self, isFull: bool, height: int):
            self.isFull = isFull
            self.height = height

    def solution2(self, head: BinaryTree):
        """
        收集子树是否是满二叉树，子树高度
        左树满 && 右树满 && 左右树高度一样 -> 整棵树是满的
        :param head:
        :return:
        """
        if head is None:
            return True
        return self.process2(head).isFull

    def process2(self, h: BinaryTree):
        if h is None:
            return IsFull.IsFullInfo2(True, 0)
        leftInfo = self.process(h.left)
        rightInfo = self.process(h.right)
        isFull = leftInfo.isFull and rightInfo.isFull and leftInfo.height == rightInfo.height
        height = max(leftInfo.height, rightInfo.height) + 1
        return IsFull.IsFullInfo2(isFull, height)


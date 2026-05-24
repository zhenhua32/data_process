"""
实现字典数, 用于前缀查找
"""

from collections import defaultdict


class Node:
    def __init__(self):
        self.children = defaultdict(Node)
        self.is_word = False
        self.word = None
        self.data = None

    def __repr__(self) -> str:
        return f"Node({self.word})"


class Trie:
    def __init__(self) -> None:
        self.root = Node()

    def insert(self, word: str, data=None) -> None:
        """
        插入数据
        """
        node = self.root
        # 遍历每个字符, 并使得节点指向下一个节点
        for c in word:
            node = node.children[c]

        node.is_word = True
        node.word = word
        node.data = data

    def search(self, word: str) -> Node:
        """
        查找数据, 找到完全匹配的, 并返回 node
        """
        node = self.root
        for c in word:
            node = node.children.get(c)
            if node is None:
                return None

        return node

    def search_prefix(self, prefix: str) -> list[Node]:
        """
        查找数据, 找到前缀匹配的, 并返回 node
        """
        node = self.root
        for char in prefix:
            node = node.children.get(char)
            if not node:
                return []
        # 前面已经遍历完前缀了
        result = []
        # 调用辅助函数
        self.dfs(node, prefix, result)
        # 按长度升序
        result.sort(key=lambda x: len(x.word))
        return result

    def dfs(self, node: Node, word: str, result: list):
        """
        搜索所有的子节点, 并将结果添加到 result 中
        """
        if node.is_word:
            result.append(node)
        for char, child in node.children.items():
            self.dfs(child, word + char, result)


if __name__ == "__main__":
    trie = Trie()
    trie.insert("百度")
    trie.insert("百度云")
    trie.insert("百度地图")
    trie.insert("百度搜索")

    print(trie.search("百度云"))
    print(trie.search("百度地"))
    print(trie.search_prefix("百度"))
    print(trie.search_prefix("百度地"))
    print(trie.search_prefix("百度地图上"))

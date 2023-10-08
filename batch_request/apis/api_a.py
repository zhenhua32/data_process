"""
定义一个接口
"""


def request_a(query: str):
    """
    请求接口, 并解析结果

    Args:
        query (str): 输入的文本
    """
    return {"result": query}


if __name__ == "__main__":
    print(request_a("hello"))

"""
定义一个接口
"""


class ApiB:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query: str) -> dict:
        return {"url": self.url, "query": query}


if __name__ == "__main__":
    api_b = ApiB("http://localhost:8000/api_b")
    print(api_b("hello"))

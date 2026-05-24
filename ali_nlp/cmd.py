import time
import json
from multiprocessing.dummy import Pool

from client import Client, params_model


class Throttle:
    def __init__(self, rate):
        self.rate = rate  # 限制速率
        self.tokens = 0  # 当前剩余的令牌数
        self.last = 0  # 上一次消费的时间

    def consume(self, amount=1):
        now = time.time()

        if self.last == 0:
            self.last = now

        # 经过的时间
        elapsed = now - self.last

        # 当前剩余的令牌数, 不能一点点增加, 会超过任意间隔的 1 秒限制
        if int(elapsed):
            self.tokens += int(elapsed * self.rate)
            self.last = now

        # 当前的令牌数, 总之不能超过限制的速率
        self.tokens = min(self.tokens, self.rate)

        if self.tokens >= amount:
            self.tokens -= amount
        else:
            amount = 0

        return amount


def test():
    start = time.time()
    client = Client(".env")
    # 实际上用 20 还是会超过 QPS 限制
    throttle = Throttle(rate=20)
    with open("test.txt", "w", encoding="utf-8") as f:
        for i in range(300):
            # 没资源的时候, 返回 0, 所以结果是 True, 所以 while 会继续
            while not throttle.consume():
                time.sleep(1)
            params = params_model.ReqGetWsChGeneral(Text="我是中国人")
            resp = client.request(params)
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")
    # 太坑了, 不知道怎么逼近极限, 现在 300 个需要 25 秒
    print(time.time() - start)


def test2():
    start = time.time()
    cur_time = time.time()
    client = Client(".env")
    rate = 20
    cur_rate = 0
    with open("test.txt", "w", encoding="utf-8") as f:
        for i in range(300):
            now = time.time()
            print(i, now)
            if now - cur_time > 1:
                cur_rate = 1
                cur_time = now
            elif cur_rate < rate and cur_time + 1 > now:
                cur_rate += 1
            else:
                # time.sleep(cur_time + 1 - now + 0.01)
                time.sleep(1)
                cur_time = time.time()
                cur_rate = 1
            params = params_model.ReqGetWsChGeneral(Text="我是中国人")
            resp = client.request(params)
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")

    # 还是个坑爹实现
    print(time.time() - start)


def test3():
    start = time.time()
    cur_time = time.time()
    client = Client(".env")
    rate = 20
    cur_rate = 0
    queue = [cur_time] * rate
    with open("test.txt", "w", encoding="utf-8") as f:
        for i in range(300):
            now = time.time()
            print(i, now)
            if now - cur_time > 1 and queue[cur_rate] + 1 < now:
                cur_rate = 1
                cur_time = now
            elif cur_rate < rate and cur_time + 1 > now and queue[cur_rate] + 1 < now:
                cur_rate += 1
            else:
                time.sleep(cur_time + 1 - now)
                # time.sleep(1)
                cur_time = time.time()
                cur_rate = 1
            # 放入当前第 N 个位置的时间
            queue[cur_rate - 1] = time.time()
            params = params_model.ReqGetWsChGeneral(Text="我是中国人")
            resp = client.request(params)
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")

    print(time.time() - start)


def test4():
    start = time.time()
    client = Client(".env")
    pool = Pool(20)
    with open("test.txt", "w", encoding="utf-8") as f:
        while True:
            cur_time = time.time()
            params = params_model.ReqGetWsChGeneral(Text="我是中国人")
            params_list = [params] * 20
            # 并发请求
            result = pool.map(client.request_without_exception, params_list)
            for resp in result:
                f.write(json.dumps(resp, ensure_ascii=False) + "\n")
            now = time.time()
            print(cur_time, now)
            # 时间不到 1 秒, 就等待, 然后开始下一次并发请求
            wait_time = 1 - (now - cur_time)
            if wait_time > 0:
                time.sleep(wait_time)


if __name__ == "__main__":
    test4()

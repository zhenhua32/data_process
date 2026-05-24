import os
import random
import time
from multiprocessing import Pool

from side import hello


def local_hello(arg):
    print("hello", os.getpid())
    wait_time = random.random() * 2
    time.sleep(wait_time)
    print("world")
    return wait_time


def main():
    # 即使你使用的是 local_hello, 使用进程池时, 依旧会在所有的进程空间中重新导入所有模块
    pool = Pool(2)
    result = pool.map(local_hello, range(4))
    print(result)


if __name__ == "__main__":
    print("main", os.getpid())
    main()

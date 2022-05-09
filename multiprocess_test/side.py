import time
import os
import random

# 有副作用的模块, 每次被导入时, 会生成一个 txt 文件

with open(f"{os.getpid()}_{time.time()}.txt", "w") as f:
    f.write(f"{os.getpid()}")


def hello(arg):
    print("hello", os.getpid())
    wait_time = random.random() * 2
    time.sleep(wait_time)
    print("world")
    return wait_time

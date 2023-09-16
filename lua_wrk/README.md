[TOC]

# 初始化

本目录包含一个容器配置, 用于在 vscode 中运行.

可以自己编译 lua 和 wrk. 同时预先编译好的放在 bin 目录下.

```bash
export PATH=/workspaces/data_process/lua_wrk/bin:$PATH
```

# 启动服务

```bash
uvicorn server:app
# 如果需要热加载, 可以用 --reload
```


```bash
curl http://127.0.0.1:8000
curl http://127.0.0.1:8000/items/1?q=10
```


# 使用 wrk 压测

```bash
wrk -t2 -c100 -d30s -R2000 http://127.0.0.1:8000
```

# hmac-sha256

sha256.lua 来自 https://github.com/jqqqi/Lua-HMAC-SHA256

# 安装 lua

[lua 官网](http://www.lua.org/)

```bash
curl -R -O http://www.lua.org/ftp/lua-5.4.4.tar.gz
tar zxf lua-5.4.4.tar.gz
cd lua-5.4.4
make all test
```

或者选择下载预先编译的二进制包, [下载地址](http://luabinaries.sourceforge.net/)

# 安装 wrk

[wrk2](https://github.com/giltene/wrk2)

[wrk](https://github.com/wg/wrk)

虽然说 wrk2 是 wrk 的改进版, 但是更早停更, 2019 年就停止更新了.

安装方式都是直接 git clone, 然后进入到目录下 make. 
看起来 wrk2 的依赖更多点, 如果遇到编译错误, 一般是缺少文件, 直接谷歌然后装依赖吧.

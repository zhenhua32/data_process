-- 定义一个全局变量 items，用于存储文件中的所有 item
local items = {}

-- 定义一个初始化函数，用于在测试开始前执行
function init(args)
  -- 打开本地文件，假设文件名为 items.txt
  local file = io.open("items.txt", "r")
  -- 如果文件存在
  if file then
    -- 遍历文件中的每一行
    for line in file:lines() do
      -- 去掉行尾的换行符
      line = line:gsub("\n", "")
      -- 将行添加到 items 列表中
      table.insert(items, line)
    end
    -- 关闭文件
    file:close()
  else
    -- 如果文件不存在，打印错误信息并退出
    print("File not found")
    os.exit()
  end
end

-- 定义一个请求函数，用于在每次发送请求前执行
function request()
  -- 从 items 列表中随机选择一个 item
  local item = items[math.random(#items)]
  -- 构建请求体
  local body = '{"name": "%s"}'
  body = body:format(item)
  -- 返回请求方法，路径，头部和体部
  wrk.headers["Content-Type"] = "application/json"

  -- format 方法会将里面的值, 和现有的 wrk table 进行合并
  return wrk.format("POST", "/items/1", nil, body)
end

-- 定义一个响应函数，用于在每次接收响应后执行
function response(status, headers, body)
  -- 打印响应状态码和内容长度
  if (status ~= 200)
  then
    print(status, headers["content-length"], body)
  else
    -- print(status, headers["content-length"], body)
  end
end


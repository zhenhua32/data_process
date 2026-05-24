import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.message import EmailMessage

from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env")

fromaddr = os.environ["fromaddr"]
toaddrs = [os.environ["toaddrs"]]
password = os.environ["password"]


"""
https://blog.csdn.net/MATLAB_matlab/article/details/106240424
"""

# 纯文本
msg = EmailMessage()
msg.set_content("Python 邮件发送测试...")
# 发送者
msg["From"] = fromaddr
# 接收者
msg["To"] = ", ".join(toaddrs)
# 主题 TODO: 这个在 QQ 邮箱中有点坑, 会吞了前面的英文部分, 估计是 EmailMessage 有问题
msg["Subject"] = "Hello 主题是邮件测试"

# 如果用 MIMEMultipart 的话, 主题是正常的, 前面的英文也能显示
msg = MIMEMultipart()
msg.attach(MIMEText("Python 邮件发送测试...", "plain", "utf-8"))
# 发送者
msg["From"] = fromaddr
# 接收者
msg["To"] = ", ".join(toaddrs)
msg["Subject"] = "Hello 主题是邮件测试"


data_list = [
    {"姓名": "小", "性别": "男", "年龄": 18},
    {"姓名": "小", "性别": "男", "年龄": 17},
    {"姓名": "小", "性别": "女", "年龄": 15},
    {"姓名": "小", "性别": "女", "年龄": 12},
]
df = pd.DataFrame(data_list)


now_time = datetime.datetime.now()
year = now_time.year
month = now_time.month
day = now_time.day
mytime = str(year) + " 年 " + str(month) + " 月 " + str(day) + " 日 "
fayanren = "爱因斯坦"
zhuchiren = "牛顿"
# 构造HTML
css = """
<style>
/* includes alternating gray and white with on-hover color */

.mystyle {
    font-size: 11pt; 
    font-family: Arial;
    border-collapse: collapse; 
    border: 1px solid silver;

}

.mystyle td, th {
    padding: 5px;
}

.mystyle tr:nth-child(even) {
    background: #E0E0E0;
}

.mystyle tr:hover {
    background: silver;
    cursor: pointer;
}
</style>
"""
content = """
<html>
<head>{css}</head>
<body>
    <h1 align="center">这个是标题，xxxx通知</h1>
    <p><strong>您好：</strong></p>
    <blockquote><p><strong>以下内容是本次会议的纪要,请查收！</strong></p></blockquote>
    <blockquote><p><strong>发言人：{fayanren}</strong></p></blockquote>
    <blockquote><p><strong>主持人：{zhuchiren}</strong></p></blockquote>

    {dataframe}

    <p align="right">{mytime}</p>
<body>
<html>
""".format(
    fayanren=fayanren, zhuchiren=zhuchiren, mytime=mytime, dataframe=df.to_html(classes="mystyle"),
    css=css,
)
msg.attach(MIMEText(content, "html", "utf-8"))


server = smtplib.SMTP_SSL("smtp.qq.com", 465)
server.login(fromaddr, password)
server.set_debuglevel(1)
server.send_message(msg)
# server.sendmail(fromaddr, ", ".join(toaddrs), msg.as_string())
server.quit()

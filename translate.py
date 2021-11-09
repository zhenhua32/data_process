from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pyperclip
import time

url = "https://translate.google.cn/?hl=zh-CN"
# 链接自己造, 自己决定从哪种语言翻译到哪种语言, sl 是源语言, tl 是目标语言
url = "https://translate.google.cn/?hl=zh-CN&sl=en&tl=zh-CN&op=translate"

service = Service(executable_path="./data/chromedriver.exe")
driver = webdriver.Chrome(service=service)


def translate(query):
    """
    进行一次翻译
    """
    # 打开网址
    driver.get(url)

    # 等待 textarea 部分可以点击复制
    wait = WebDriverWait(driver, 10)
    sl_element = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "textarea")))
    sl_element.click()

    # 将 query 复制进去
    sl_element.send_keys(query)

    # 翻译是会自动进行的, 等待翻译完成
    tl_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "J0lOec")))
    return tl_element.text


def translate_file(input_file: str, output_file: str, text_length: int = 1000):
    """翻译单个文件

    Args:
        input_file (str): 输入文件的路径
        output_file (str): 输出文件的路径
        text_length (int): 每次翻译的文本长度, 最高不能超过 5000
    """
    with open(input_file, "r", encoding="utf-8") as f:
        with open(output_file, "w", encoding="utf-8") as fw:
            text = ""
            for line in f:
                # 我要保证单次翻译字符不能超出特定限制, 所以取了个百分比
                if len(text) > text_length * 0.9:
                    fw.write(translate(text).strip() + "\n")
                    text = line
                else:
                    text += line

            # 最后, 如果还有剩余的 text
            if text:
                fw.write(translate(text).strip() + "\n")


if __name__ == "__main__":
    url = "https://translate.google.cn/?hl=zh-CN&sl=zh-CN&tl=en&op=translate"

    query = """
连衣裙
真丝连衣裙
女装
真丝香云纱连衣裙
桑蚕丝a字裙大牌
女连衣裙
真丝女装
长裙
重磅真丝连衣裙大牌欧美时尚
的女装
桑蚕丝连衣裙
    """
    # print(translate(query))

    translate_file("./data/query.txt", "./data/query_result.txt")
    driver.quit()

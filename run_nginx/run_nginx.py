import os
import time
import requests


os.system("sudo nginx -s stop")
os.system("sudo nginx -c /home/tzh/code/nginx.conf")

time.sleep(5)

def get_result():
    url = "http://localhost:8123"
    response = requests.get(url)
    print(response.text)

    print("----\n")
    url = "http://127.0.0.1:8123"
    response = requests.get(url)
    print(response.text)


if __name__ == "__main__":
    # curl localhost:8123
    get_result()

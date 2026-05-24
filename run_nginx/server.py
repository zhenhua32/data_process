import time
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/get", methods=["GET"])
def get_example():
    return jsonify(message="This is a GET request")


@app.route("/get/long", methods=["GET"])
def get_long_example():
    time.sleep(10)
    return jsonify(message="This is a GET request")


@app.route("/post", methods=["POST"])
def post_example():
    data = request.get_json()
    return jsonify(message="This is a POST request", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""
curl -X GET http://localhost:5000/get
curl -X POST http://localhost:5000/post -H "Content-Type: application/json" -d '{"key": "value"}'
"""

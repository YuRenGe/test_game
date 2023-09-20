import sys
from flask import Flask, request, jsonify

from online_run import Env

app = Flask(__name__)

env = None


@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json()
    command = callback(data)
    resp = jsonify(command)
    return resp


def callback(json_data):
    env.update_env(json_data)
    env.construct_target_pos()
    res = env.run()
    return res


if __name__ == '__main__':
    env = Env()
    app.run(port=int(sys.argv[1]))

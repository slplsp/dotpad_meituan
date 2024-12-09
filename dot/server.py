from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("dot.html")


@app.route('/run-python-script', methods=['POST'])
def run_script():
    # 파이썬 스크립트 실행
    os.system('python file.py')
    return jsonify({'message': 'Script executed successfully!'})


# @app.route('/run_script')
# def run_a_script():
#     try:
#         # a.py가 있는 경로로 변경해야 할 수도 있습니다.
#         result = subprocess.run(['python', 'static/bbox.py'], stdout=subprocess.PIPE)
#         # 스크립트의 출력을 반환합니다.
#         return result.stdout.decode('utf-8')
#     except Exception as e:
#         return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    import subprocess
    subprocess.check_output("python3 demo.py", shell=1)
    return "<p>Hello, World!</p>"
if __name__ == '__main__':
   app.run(debug=True, port=8080)
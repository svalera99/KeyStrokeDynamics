from flask import Flask, render_template, request
import os
app = Flask(__name__)

@app.route("/", methods=["GET"])
def render():
    return render_template("index.html")

@app.route("/", methods=["GET","POST"])
def func():
    if request.method == "POST":
        username = request.form["main_input"]
        print(username)

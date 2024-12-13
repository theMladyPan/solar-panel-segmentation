#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)


@app.route("/")
def index():
    # route to home page
    return render_template("index.j2", active_page="home")


@app.route("/home")
def home():
    return render_template("index.j2", active_page="home")


@app.route("/files")
def files():
    # for each file in uploads directory, add to list
    files = os.listdir("uploads")
    # get metadata for each file
    files = [{"filename": file, "size": int(os.path.getsize(f"uploads/{file}") / 1024)} for file in files]

    return render_template("files.j2", active_page="files", files=files)


@app.route("/uploaded_file")
def uploaded_file():
    filename = request.args.get("filename")
    # serve the file
    return send_from_directory("uploads", filename)


@app.route("/stitch")
def stitch():
    return render_template("stitch.j2", active_page="stitch")


@app.route("/crop")
def crop():
    return render_template("crop.j2", active_page="crop")


@app.route("/analyze")
def analyze():
    return render_template("analyze.j2", active_page="analyze")


@app.route("/upload", methods=["POST"])
def upload():
    # get all files from the request
    uploaded_files = request.files.getlist("file")
    print(uploaded_files)
    # loop through all files
    for file in uploaded_files:
        # save the file to the server
        file.save(f"uploads/{file.filename}")
        print(f"uploads/{file.filename}")
    # return a response
    return redirect(url_for("files"))


if __name__ == "__main__":
    app.run(debug=True)


import os, glob, sys, pickle, random, csv, threading
import numpy as np

from flask import Flask, request, redirect, url_for
from flask import render_template, url_for, send_file
from flask import send_from_directory
from werkzeug.utils import secure_filename

from functools import wraps, update_wrapper
from datetime import datetime

from styler import transfer


UPLOAD_FOLDER = '/home/nik/Projects/image-prior/sketch-playground/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/image')
def random_normal():
    return send_file("patch.png")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print ("here")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            args = ("images/wave.jpg", path, "uploads/out.jpg")
            thread = threading.Thread(target=transfer, args=args)
            thread.start()
            return redirect(url_for('uploaded_file',
                                    filename='out.jpg'))
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=file>
             <input type=submit value=Upload>
        </form>
        ''' 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


app.run(debug=True)
import os
import numpy as np
import cv2 as cv
from flask import Flask, request, render_template, redirect, url_for, flash

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

pattern_size = (7, 6)  # internal corners
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecret'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            results = calibrate_from_image(filepath)
            if results is None:
                flash('Checkerboard NOT found in the image!')
                return redirect(request.url)
    return render_template('index.html', results=results)

def calibrate_from_image(image_path):
    # Prepare object points
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    img = cv.imread(image_path)
    if img is None:
        return None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
    else:
        return None

    # Calibrate (single image)
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return {
        'K': mtx,
        'dist': dist,
        'reproj_error': ret
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

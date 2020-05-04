import os
import cv2
import numpy as np
import urllib.parse
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from demo_pipeline import Pipeline
from facenet import facenet_classifier as fc# TODO: fix path facenet/facenet_classifier
import imutils
import time
from utils import prepare_batch, unzip_and_move, get_files, copy_originals, number_of_files, pretty_print
from universum_api import post_batch, get_status, get_processed

pipeline = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['zip', 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = b'_5#y2Lpieterisawesome\xec]/haha'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def np_array_to_string(np_array):
    plt.axis('off')
    plt.imshow(np_array)
    buffered = BytesIO()
    plt.savefig(buffered, format='png', dpi=100)
    img_str = base64.b64encode(buffered.getvalue())
    img_str = urllib.parse.quote(img_str)
    return img_str

def upload_to_numpy(file):
    filestr = request.files['file'].read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def np_array_to_string(np_array):
    plt.axis('off')
    plt.imshow(np_array)
    buffered = BytesIO()
    plt.savefig(buffered, format='png', dpi=100)
    img_str = base64.b64encode(buffered.getvalue())
    img_str = urllib.parse.quote(img_str)
    return img_str


def upload_to_numpy(file):
    filestr = request.files['file'].read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    MAX_WIDTH = 1000
    img = imutils.resize(img, width=MAX_WIDTH)


    return img

def add_target(img):
    # generate augmentations
    poses = [
        {
            "yaw": -45,
            "pitch": 0,
            "roll": 0
        },
        {
            "yaw": -30,
            "pitch": 0,
            "roll": 0
        },
        {
            "yaw": -15,
            "pitch": 0,
            "roll": 0
        },
        {
            "yaw": 15,
            "pitch": 0,
            "roll": 0
        },
        {
            "yaw": 30,
            "pitch": 0,
            "roll": 0
        },
        {
            "yaw": 45,
            "pitch": 0,
            "roll": 0
        }
    ]

    folder_name = str(time.time())
    os.mkdir("./tmp/" + folder_name)

    cv2.imwrite("./tmp/" + folder_name + "/" + "original.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    bundle_loc = "./tmp/" + folder_name + ".zip"
    prepare_batch(["./tmp/" + folder_name + "/" + "original.png"], bundle_loc)
    process_id = post_batch(poses=poses, bundle_loc=bundle_loc)

    processed_loc = "./tmp/" + folder_name + "_processed.zip"

    time.sleep(3)
    finished = False
    while not finished:
        finished = get_processed(process_id, processed_loc)  # downloads the zip
        time.sleep(5)

    unzip_and_move(processed_loc, "./tmp/" + folder_name + "/", ["henk"])

    img_list = []
    for file in sorted(os.listdir("./tmp/" + folder_name))[::-1]:
        img = cv2.cvtColor(cv2.imread("./tmp/" + folder_name + "/" + file), cv2.COLOR_BGR2RGB)
        pipeline.add_to_targets(img)
        img_list.append(np_array_to_string(img))
        os.remove("./tmp/" + folder_name + "/" + file)

    os.rmdir("./tmp/" + folder_name)


    return img_list

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # read image file string data
            img = upload_to_numpy(file)
            img_str = np_array_to_string(img)
            if request.form["button"] == "upload target":
                img_list = add_target(img)
                return render_template("show_all_images.html", images=img_list)

            new_img = pipeline.pipeline(img)
            new_img_str = np_array_to_string(new_img)

            return render_template("show_images.html", original=img_str, processed=new_img_str)
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
    return render_template("upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    temporary_location = "target_embeddings"
    temporary_csv = "temporary_embeddings.csv"
    temporary_path = os.path.join(temporary_location, temporary_csv)
    print(temporary_path)
    os.makedirs("tmp", exist_ok=True)
    classifier = fc.FacenetClassifier(location=temporary_path, threshold=0.8)
    pipeline = Pipeline(classifier)
    bootstrap = Bootstrap(app)
    app.run(host='0.0.0.0', port=8888, debug=True)

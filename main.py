
from flask import Flask, render_template, request, json, jsonify, session, redirect, send_file, url_for, flash
import plotly.graph_objs as go
from matplotlib.pylab import *
import pylab
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt

print("Starting..")

# import systemcheck

# import the necessary packages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Take a look at a iceberg

IMAGE_SIZE = (75, 75, 3)

CATEGORIES = {1: 'No-Iceberg(Ship)', 0: 'Iceberg'}


def detect_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        print("Image loaded successfully")
        # Apply a threshold to the image to separate the iceberg from the background
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Find the contours of the iceberg
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw bounding boxes around the detected icebergs and annotate their width
        widths = []
        for c in contours:
            # Get the coordinates of the bounding box
            x, y, w, h = cv2.boundingRect(c)
            # Draw the bounding box on the original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Calculate the width of the iceberg and annotate it on the image
            width = w
            widths.append(width)
            # cv2.putText(img, f"{width}", (x, y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            print("Iceberg width:", width)

        # Resize the image for display
        output_img = cv2.resize(img, (500, 500))
        # Encode the image in base64 format for rendering in HTML
        buffer = BytesIO()
        plt.imsave(buffer, output_img, format='png')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        # Return the encoded image file and widths of each box
        return f"data:image/png;base64,{img_str}", widths
    else:
        print("Failed to load image")
        return None


def plotmy3d(c, name):

    data = [go.Surface(z=c)]
    layout = go.Layout(title=name, autosize=False, width=700,
                       height=700, margin=dict(l=65, r=50, b=65, t=90))
    fig = go.Figure(data=data, layout=layout)
    fig.write_html("static/uploads/file2.html")
    # py.iplot(fig)


model = load_model("icebergdetection_CNN.h5")


def model_warmup():
    dummy_image = []
    for i in range(IMAGE_SIZE[0]):
        dummy_image.append([[0]*3]*IMAGE_SIZE[1])
    image = np.array(dummy_image)
    # print(image.shape)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    # print(pred)


def predict_disease(imgpath):
    start_time = time.time()
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)

    img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_hor = cv2.flip(img, 1)

    images = []
    images.append(img)
    images.append(img_rotated_90)
    images.append(img_rotated_180)
    images.append(img_rotated_270)
    images.append(img_flip_ver)
    images.append(img_flip_hor)

    images = np.array(images)
    images = images.astype(np.float32)
    images /= 255

    op = []
    prob = []
    # make predictions on the input image
    for im in images:
        image = np.array(im)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_prob = np.max(pred, axis=1)[0]
        op.append(pred_class)
        prob.append(pred_prob)

        print("Predicted Class:",
              CATEGORIES[pred_class], "with probability:", pred_prob)

    op = np.array(op)
    final_pred_class = CATEGORIES[np.bincount(op).argmax()]
    final_pred_prob = np.max(np.bincount(op))/len(op)
    print("Final Output:", final_pred_class,
          "with probability:", final_pred_prob)
    end_time = time.time()  # End time
    total_time = round(end_time - start_time, 2)  # Total time
    return final_pred_class, round(final_pred_prob, 2), total_time


model_warmup()

app = Flask(__name__)
app.secret_key = "secure"
app.config['UPLOAD_FOLDER'] = str(os.getcwd())+'/static/uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=["post", "get"])
def first_page():
    if request.method == "POST":
        global image_name, image_data

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            op, percentage, graph = predict_disease('static/uploads/'+filename)

            # Write you 3D Plot Code here and save it as "file2.html", No need to plot here just save
            # Make Sure to save it in /static/uploads folder
            img = cv2.imread('static/uploads/'+filename)

            # print(img[:,:,0].shape)
            plotmy3d(img[:, :, 0], str(op))

            # solution = SOLUTIONS[op]
            # Path to the input image file
            image_path = filepath
            # Detect icebergs in the image and return the encoded output image
            output_image, widths = detect_image(image_path)

            html_3d_file = "file2.html"
            return render_template("data_page.html",
                                   filename=filename, file_3d=html_3d_file, result=op, percentage=percentage, graph=graph,
                                   output_image=output_image, widths=widths)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif,tif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


app.run(debug=True, host="0.0.0.0")

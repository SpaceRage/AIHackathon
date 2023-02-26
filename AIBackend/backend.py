import io
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory
from flask_restful import Resource, Api, request
import base64
from PIL import Image
import json
import pyperclip
from io import BytesIO
import cv2 as cv
from flask_cors import CORS


app = Flask(__name__)
api = Api(app)
CORS(app)


@app.route('/')
def home():
    return send_from_directory('website', 'index.html')

@app.route('/<path:path>')
def send_image(path):
    return send_from_directory('website', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    img_data = request.files['file'].read()

    # Get the JSON data from the request
    json_data = request.form.to_dict()

    # Decode the image data and create a PIL Image object
    img = Image.open(BytesIO(base64.b64decode(base64.b64encode(img_data))))
    # save the image
    img.save('test.jpg')

    # Process the image and get the prediction
    prediction = useModel(process_image('test.jpg'), json_data['age'], json_data['sex'], json_data['localization'])
    print(prediction)

    # Return the prediction as a JSON response
    return {'data': str(prediction)}, 200
        


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def inBounds(coord, dim):
    return coord > dim/3 and coord < dim/3 * 2

def process_image(filename):
    # open image
    file = cv.imread(filename)
    imgcolor = file
    kernel = np.ones((10, 10), np.uint8)

    shave = cv.morphologyEx(imgcolor, cv.MORPH_CLOSE, kernel, iterations=2)
    blur = cv.blur(shave, (10, 10))

    img = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    dst = crop_img(cv.equalizeHist(img), 0.75)
    (T, threshInv) = cv.threshold(dst, 40, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = cv.findContours(
        threshInv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return
    blob = max(contours, key=lambda el: cv.contourArea(el))
    #blob = sorted(contours, key=lambda el: cv.contourArea(el))
    M = cv.moments(blob)
    if M["m00"] == 0:
        return
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # print(center)
    (x, y, w, h) = cv.boundingRect(blob)
    dim = dst.shape
    if x - 200 > 0:
        x = x - 200
    else:
        x = 0
    if y - 200 > 0:
        y = y-200
    else:
        y = 0
    if w + 200 < dim[0]:
        w = w+200
    else:
        w = dim[0]
    if h + 200 < dim[1]:
        h = h+200
    else:
        h = dim[1]

    mask = np.zeros_like(dst)
    cv.drawContours(mask, contours, -1, (255, 255, 255), cv.FILLED)
    cv.drawContours(mask, blob, -1, (255, 255, 255), 400)
    result = cv.bitwise_and(dst, mask)
    blackMask = cv.inRange(result, 0, 10)
    percent_black = cv.countNonZero(blackMask)/result.size
    include = False
    if (percent_black * 100) < 97 and inBounds(center[0], img.shape[0]):
        include = True
    #stencil = np.zeros(dst.shape).astype(dst.dtype)
    #cv.fillPoly(stencil, blob, [255, 255, 255])
    #result = cv.bitwise_and(dst, stencil)
    #cv.drawContours(dst, scale_contour(blob, 2), -1, (255, 255, 255), 20)
    #cv.circle(dst, center, 10, (255, 255, 255), 10)
    # save the image
    cv.imwrite('test.jpg', result)

def useModel(image, age, sex, localization):
    # Load the saved model
    model = tf.keras.models.load_model('test.h5')
    img = Image.open('test.jpg')
    # Preprocess the image
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # rescale image to 0-1
    img = np.repeat(img, 3, axis=2)
    img = np.array([img], dtype='float16')

    # Define input values for prediction
    age = np.array([age], dtype='float16')
    sex = np.array([sex], dtype='float16')
    localization = np.array([localization], dtype='float16')

    # Make a prediction
    prediction = model.predict([img, age, sex, localization])

    # Return the predicted class
    return np.argmax(prediction)


if __name__ == '__main__':
    app.run() 

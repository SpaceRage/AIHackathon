import io
import tensorflow as tf
import numpy as np
from flask import Flask
from flask_restful import Resource, Api, request
import base64
from PIL import Image

app = Flask(__name__)
api = Api(app)


class Model(Resource):
    def post(self):
        data = request.data
        img_data = base64.b64decode(data)
        #Image.open(io.BytesIO(img_data)).save('img.png')
        print(data)
        

def useModel(img):
    # Load the saved model
    model = tf.keras.models.load_model('test.h5')

    # Preprocess the image
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # rescale image to 0-1
    img = np.array([img], dtype='float16')

    # Define input values for prediction
    test_age = np.array([25], dtype='float16')
    test_sex = np.array([0], dtype='float16')
    test_localization = np.array([1], dtype='float16')

    # Make a prediction
    prediction = model.predict([img, test_age, test_sex, test_localization])

    # Return the predicted class
    return np.argmax(prediction)


api.add_resource(Model, '/model')

if __name__ == '__main__':
    app.run() 

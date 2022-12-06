from flask import Flask, render_template, request, redirect, url_for, jsonify
# import base64
# import cv2
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)

# print(tf.__version__)

# model = tf.keras.models.load_model('keras.h5')
# model.make_predict_function()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recognize', methods=['GET', 'POST'])
# def recognize():
    
#     if request.method == 'POST':

#         data = request.get_json()
#         imageBase64 = data['image']
#         imgBytes = base64.b64decode(imageBase64)

#         with open("temp.jpg", "wb") as temp:
#             temp.write(imgBytes)

#         with open("class_names.txt") as f:
#             class_names = f.readlines()

#         classes = [c.replace('\n', '').replace(' ', '_') for c in class_names]
#         img = cv2.imread("temp.jpg")
#         img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         img_prediction = np.reshape(img_gray, (1, 28, 28, 1))
#         img_prediction = (255-img_prediction.astype('float'))/255.0

#         prediction = model.predict(np.expand_dims(img, axis=0))[0]
#         ind = (-prediction).argsort()[:5]
#         latex = [classes[x] for x in ind]

#         return jsonify({
#             'prediction': str(latex),
#             'status' : True
#         })

# if __name__ == '__main__':
#     app.run(debug=True)
import base64
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# print(tf.__version__)

model = tf.keras.models.load_model('keras.h5')
model.make_predict_function()


@app.route('/')
def index():

    return render_template('index.html')

@app.route('/recognize', methods = ['POST'])
def recognize():

    if request.method == 'POST':
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)

        with open('class_names.txt') as f:
            class_name = f.readlines()
        classes = [c.replace('\n', '').replace(' ', '_') for c in class_name]

        image = cv2.imread('temp.jpg')
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_prediction = np.reshape(image_gray, (28,28,1))
        image_prediction = (255 - image_prediction.astype('float')) / 255

        # prediction = np.argmax(model.predict(np.array([image_prediction])), axis = -1)

        prediction = model.predict(np.expand_dims(image_prediction, axis=0))[0]
        ind = (-prediction).argsort()[:1]
        latex = [classes[x] for x in ind][0]
    return jsonify({
        'prediction' : str(latex),
        'status' : True
    })

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug = True)

    
from flask import Flask, render_template, request
import pickle
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

app = Flask(__name__)

# Load the model
model = pickle.load(open("model/random_forest_model.pkl", "rb"))

# Class labels
classes = ['Healthy', 'BlackSpot', 'Anthracnose']

# Feature extractors
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg16.layers: layer.trainable = False
for layer in vgg19.layers: layer.trainable = False

gap16 = GlobalAveragePooling2D()(vgg16.output)
gap19 = GlobalAveragePooling2D()(vgg19.output)

model16 = Model(inputs=vgg16.input, outputs=gap16)
model19 = Model(inputs=vgg19.input, outputs=gap19)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)

            feat16 = model16.predict(image)
            feat19 = model19.predict(image)

            features = np.hstack((feat16, feat19))

            pred = model.predict(features)
            prediction = classes[int(pred[0])]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

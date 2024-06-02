from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Modell laden
model = load_model('./model/deepfake-detection-v2.keras')

# Funktion zur Vorhersage
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        result = 'Deepfake' if prediction > 0.5 else 'Real'
        return render_template('index.html', prediction=result, image_path=file_path)
    return redirect(request.url)

@app.route('/logs')
def logs():
    logs_path = './logs/tensorboard_logs'
    os.system(f'tensorboard --logdir={logs_path} --host=0.0.0.0 --port=6006')
    return redirect("http://localhost:6006")

if __name__ == '__main__':
    app.run(debug=True)

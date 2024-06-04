import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
import base64

def startTensorboard(logs_path):
    os.system(f'tensorboard --logdir={logs_path} --host=0.0.0.0 --port=6006')

def loadModel(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    return model

def predict_image(image, model):
    img = Image.open(BytesIO(image)).resize((256, 256))  # Adjust target size as needed
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    prediction = model.predict(np.expand_dims(img_array / 255.0, 0))  # Ensure image is normalized
    return prediction[0][0]

def resizeImage(image):
    resize_image = tf.image.resize(image, (256, 256))
    return resize_image

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model/deepfake-detection-v2.keras')
logs_path = os.path.join(base_dir, 'logs/tensorboard_logs/')

print("Model path:", model_path)
print("Logs path:", logs_path)



# Load the model
model = loadModel(model_path)

app = Flask(__name__)

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
        file_content = file.read()
        prediction = predict_image(file_content, model)  # Pass the model as an argument
        result = prediction
        
        # Encode the image in base64
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        return render_template('index.html', prediction=result, image_data=image_data)
    return redirect(request.url)

@app.route('/tensorboard')
def tensorboard():
    startTensorboard(logs_path)

@app.route('/logs')
def logs():
    host = request.host.split(':')[0]
    tensorboard_url = f"http://{host}:6006"
    return redirect(tensorboard_url)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)

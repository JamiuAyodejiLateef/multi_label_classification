
from flask import Flask, request, render_template, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


model = load_model('best_vgg16_withregdropout.keras')

labels = ['Background', 'Crack', 'Spalling', 'Efflorescence', 'ExposedBars', 'Corrosionstain']


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    img_array = prepare_image(file_path)
    predictions = model.predict(img_array)
    predicted_labels = (predictions > 0.5).astype(int)
    
    
    present_labels = [label for label, prediction in zip(labels, predicted_labels[0]) if prediction == 1]

    
    img = image.load_img(file_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    plt.title('Predicted Classes')

    
    if present_labels:
        y_positions = np.linspace(0.08, 0.95, len(present_labels))  

        for i, (label, y) in enumerate(zip(present_labels, y_positions)):
            ax.text(1.05, y, label, transform=ax.transAxes, fontsize=12,
                    ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7, edgecolor='black'))

    
    base_name = os.path.splitext(file.filename)[0]
    output_filename = f"{base_name}_predicted.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    plt.savefig(output_path, format='jpg', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)



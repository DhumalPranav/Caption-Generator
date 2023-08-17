from flask import Flask, request, render_template, send_file
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import io
import os
if request.method == 'POST':
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_path = file_path
        max_length = 32
        tokenizer = load(open("tokenizer.p","rb"))
        model = load_model('models/model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")
        photo = extract_features(img_path, xception_model)
        img = Image.open(img_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        output_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        img.save(output_img_path)
        return render_template('index.html', output=description, image_output=output_img_path)


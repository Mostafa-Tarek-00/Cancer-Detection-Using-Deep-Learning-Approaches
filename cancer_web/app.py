import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import io
import cv2
from tensorflow.keras import backend as K

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'secret_key'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50))
    lastname = db.Column(db.String(50))
    country = db.Column(db.String(50))
    city = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        country = request.form['country']
        city = request.form['city']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'})

        # Create a new user
        new_user = User(firstname=firstname, lastname=lastname, country=country, city=city, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        # return jsonify({'message': 'Sign up successful!'})

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                session['user_id'] = user.id
                session['firstname'] = user.firstname
                return redirect(url_for('index'))
            else:
                return jsonify({'error': 'Incorrect password'})
        else:
            return jsonify({'error': 'User not found'})

    return render_template('login.html')

# Model architecture
def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_model()

model.load_weights('model_weights_new.h5')

lesion_type_dict = {
    'nv': 'Melanocytic nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign keratosis-like lesions (bkl)',
    'bcc': 'Basal cell carcinoma (bcc)',
    'akiec': 'Actinic keratoses (akiec)',
    'vasc': 'Vascular lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}

label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

def predict_skin(image):
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    img = np.array(image)[:, :, :3]
    img = tf.image.resize(img, (28, 28))
    img = img.numpy()  
    img = img.reshape((1, 28, 28, 3))
    predictions = model.predict(img)
    max_index = np.argmax(predictions)
    predicted_label = label_mapping[max_index]
    confidence = predictions[0, max_index]
    return lesion_type_dict[predicted_label], confidence

im_width = 256
im_height = 256
smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)
    

brain_model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

def predict_brain(file):
    img = Image.open(file)
    img_array = np.array(img)

    slice_index = img_array.shape[2] // 2
    img_slice = img_array[:, :, slice_index]

    img_slice = cv2.resize(img_slice, (im_height, im_width))
    img_slice = img_slice / np.max(img_slice)

    img_slice_rgb = np.stack((img_slice,) * 3, axis=-1)

    img_slice_rgb = img_slice_rgb[np.newaxis, :, :, :]

    pred_brain = brain_model.predict(img_slice_rgb)

    alpha = 0.5
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(img_array)  
    mask_img[np.squeeze(pred_brain) > 0.5] = 255
    overlay_img = cv2.addWeighted(img_array, 1 - alpha, mask_img, alpha, 0)

    overlay_image_path = 'static/overlay_image.jpg' 
    cv2.imwrite(overlay_image_path, overlay_img)

    return overlay_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict_image():
    file = request.files['image']
    if file:
        if file.filename.endswith('.tif') or file.filename.endswith('.tiff'):
            overlay_image_path = predict_brain(file)
            return jsonify({'overlay_image': overlay_image_path})
        
        elif file.filename.endswith('.jpg') or file.filename.endswith('.jpeg'):
            image = Image.open(io.BytesIO(file.read()))
            label, confidence = predict_skin(image)
            confidence_value = confidence.item()
            return jsonify({'prediction': label, 'confidence': confidence_value})
        else:
            return jsonify({'error': 'Unsupported file format'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)

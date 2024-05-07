import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model  
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import io

# Initialize Flask app
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

        # Check if the email is already registered
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

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()
        if user:
            # Check if the provided password matches the stored password
            if user.password == password:
                # Store user information in the session
                session['user_id'] = user.id
                session['firstname'] = user.firstname
                # Redirect to the homepage or any other desired page
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

# Load model parameters
model.load_weights('model_weights_new.h5')

# Dictionary for labels
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

# Preprocess the image and make a prediction
def predict(image):
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    img = np.array(image)[:, :, :3]
    # Resize the image to 28x28 pixels
    img = tf.image.resize(img, (28, 28))
    img = img.numpy()  # Convert EagerTensor to NumPy array
    img = img.reshape((1, 28, 28, 3))
    predictions = model.predict(img)
    max_index = np.argmax(predictions)
    predicted_label = label_mapping[max_index]
    confidence = predictions[0, max_index]
    return lesion_type_dict[predicted_label], confidence

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])

def predict_image():
    file = request.files['image']
    if file:
        image = Image.open(io.BytesIO(file.read()))
        label, confidence = predict(image)
        confidence_value = confidence.item()
        return jsonify({'prediction': label, 'confidence': confidence_value})
    else:
        return jsonify({'error': 'No image provided'})

# Run the Flask app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)

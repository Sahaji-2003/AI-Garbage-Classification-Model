# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import numpy as np
# import io

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load your pre-trained Keras model
# model = load_model("model_55.keras")

# # Assuming your model expects images of a certain size
# img_size = (224, 224)

# @app.route('/classify-waste', methods=['POST'])
# def classify_waste():
#     try:
#         # Receive the image from the request
#         img_file = request.files['image']
#         img = Image.open(io.BytesIO(img_file.read()))
#         img = img.convert('RGB')  # Ensure the image has RGB channels
#         img = img.resize(img_size)
#         img_array = np.asarray(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array.astype('float32') / 255.0  # Normalize the image data

#         # Perform prediction using the loaded model
#         result = model.predict(img_array)

#         # Convert the prediction result to a human-readable class (replace this with your actual mapping)
#         classes = ['Biomedical', 'Electrical', 'Organic','Recyclable']   # Replace with your class labels
#         predicted_class = classes[np.argmax(result)]

#         return jsonify({'result': predicted_class})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pre-trained Keras model
model = load_model("model_55.keras")

# Assuming your model expects images of a certain size
img_size = (224, 224)

# Replace these labels with your actual class labels
classes = ['Biomedical', 'Electrical', 'Organic', 'Recyclable']

@app.route('/classify-waste', methods=['POST'])
def classify_waste():
    try:
        # Receive the image from the request
        img_file = request.files['image']
        img = Image.open(io.BytesIO(img_file.read()))
        img = img.convert('RGB')  # Ensure the image has RGB channels
        img = img.resize(img_size)
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0  # Normalize the image data

        # Perform prediction using the loaded model
        result = model.predict(img_array)

        # Calculate individual probabilities for each class
        probabilities = {label: round(prob * 100, 2) for label, prob in zip(classes, result[0])}

        # Convert the prediction result to a human-readable class
        predicted_class = classes[np.argmax(result)]

        # Get the maximum probability
        max_probability = np.max(result)

        return jsonify({
            'result': predicted_class,
            'max_probability': float(max_probability),
            'individual_probabilities': probabilities,
            'classes': classes,
            # 'probabilities': prob,

        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

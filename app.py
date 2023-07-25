from flask import Flask, request
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained kNN model
knn_model = joblib.load('server\model\knn_model.pkl')

# Mapping of class labels to information
class_info = {
    'Green Coffee': 'For Earthy Flavor Coffee',
    'Begin To Pale': 'Light Level Coffee',
    'Early Yellow': 'Light Level Coffee ',
    'Yellow-Tan': 'Yellowish-Gold River Cofee',
    'Light Brown': 'Medium Level Coffee',
    'Brown': 'Medium Level Coffee',
    '1st Crack Start': 'Medium Level Coffee',
    '1st Crack Done': 'Medium Level Coffee',
    'City Roast': 'Dark Level Coffee-For Average Coffee Drinker',
    'City+': 'Dark Level Coffee',
    'Full City+ 2nd Crack': 'Acidity Replaced by Smooth Sweetness',
    'Vienna Light': 'Full Bodied, Low Acidity, Heavy Mouthfeel',
    'Full French': 'Dark Brown and Oily',
    'Charcoal Dead': 'Extra Dark Level Coffee',
    'Fire Risk': 'Extra Dark Level Coffee',
    'Green Beans': 'Raw Coffee Beans',
}

@app.route('/classify', methods=['POST'])
def classify_image():
    
    # Receive the image from the Flutter app
    image = request.files['image']
    # Perform HSV image preprocessing
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    # Extract features from the preprocessed image
    feature1 = np.median(img_hsv[:, :, 0])  # Mean hue
    feature2 = np.median(img_hsv[:, :, 1])  # Mean saturation
    feature3 = np.median(img_hsv[:, :, 2])  # Mean value
    # Prepare the features for classification
    features = [[feature1, feature2, feature3]]
    # Classify the image using the kNN model
    prediction = knn_model.predict(features)
    class_label = str(prediction[0])
    app.logger.info(class_label)
    
    # Get the information about the class label
    class_information = class_info.get(class_label, 'Unknown class')
    # Return the classification result and information as a response
    return {
        'result': class_label,
        'information': class_information,
    }

if __name__ == '__main__':
    app.run()
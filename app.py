from flask import Flask, request
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained kNN model
knn_model = joblib.load('knn_model.pkl')

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

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/classify', methods=['POST'])
def classify_image():
    
    # Receive the image from the Flutter app
    image = request.files['image']
    # Perform HSV image preprocessing
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    # Extract features from the preprocessed image
    # Compute mean of each channel
    feature_mean_hue = np.mean(img_hsv[:, :, 0])
    feature_mean_saturation = np.mean(img_hsv[:, :, 1])
    feature_mean_value = np.mean(img_hsv[:, :, 2])

    # Compute median of each channel
    feature_median_hue = np.median(img_hsv[:, :, 0])
    feature_median_saturation = np.median(img_hsv[:, :, 1])
    feature_median_value = np.median(img_hsv[:, :, 2])

    # Compute 25th and 75th percentiles for each channel
    feature_percentile_25_hue = np.percentile(img_hsv[:, :, 0], 25)
    feature_percentile_25_saturation = np.percentile(img_hsv[:, :, 1], 25)
    feature_percentile_25_value = np.percentile(img_hsv[:, :, 2], 25)
    feature_percentile_75_hue = np.percentile(img_hsv[:, :, 0], 75)
    feature_percentile_75_saturation = np.percentile(img_hsv[:, :, 1], 75)
    feature_percentile_75_value = np.percentile(img_hsv[:, :, 2], 75)

    # Compute standard deviation of each channel
    feature_std_hue = np.std(img_hsv[:, :, 0])
    feature_std_saturation = np.std(img_hsv[:, :, 1])
    feature_std_value = np.std(img_hsv[:, :, 2])

    # Compute variance of each channel
    feature_var_hue = np.var(img_hsv[:, :, 0])
    feature_var_saturation = np.var(img_hsv[:, :, 1])
    feature_var_value = np.var(img_hsv[:, :, 2])

    return [feature_mean_hue, feature_mean_saturation, feature_mean_value,
            feature_median_hue, feature_median_saturation, feature_median_value,
            # *hist_hue.tolist(), *hist_saturation.tolist(), *hist_value.tolist(),
            feature_percentile_25_hue, feature_percentile_25_saturation, feature_percentile_25_value,
            feature_percentile_75_hue, feature_percentile_75_saturation, feature_percentile_75_value,
            feature_std_hue, feature_std_saturation, feature_std_value,
            feature_var_hue, feature_var_saturation, feature_var_value]
    
    # Prepare the features for classification
    features = [[feature_mean_hue, feature_mean_saturation, feature_mean_value,
            feature_median_hue, feature_median_saturation, feature_median_value,
            # *hist_hue.tolist(), *hist_saturation.tolist(), *hist_value.tolist(),
            feature_percentile_25_hue, feature_percentile_25_saturation, feature_percentile_25_value,
            feature_percentile_75_hue, feature_percentile_75_saturation, feature_percentile_75_value,
            feature_std_hue, feature_std_saturation, feature_std_value,
            feature_var_hue, feature_var_saturation, feature_var_value]]
    
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
    app.run(debug=True, host='0.0.0.0', port=80)

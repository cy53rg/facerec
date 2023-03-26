import cv2
import numpy as np
from facerec.models.cnn_model import load_cnn_model
from facerec.models.knn_model import load_knn_model
from facerec.models.svm_model import load_svm_model
from facerec.training.data_preprocessing import preprocess_image

def recognize_image(image_path, model_type='cnn', model_path=None):
    # Load input image
    image = cv2.imread(image_path)

    # Preprocess input image
    preprocessed_image = preprocess_image(image)

    # Load trained model
    if model_type == 'cnn':
        model = load_cnn_model(model_path)
    elif model_type == 'knn':
        model = load_knn_model(model_path)
    elif model_type == 'svm':
        model = load_svm_model(model_path)

    # Make prediction on preprocessed image using loaded model
    if model_type == 'cnn':
        prediction = model.predict(preprocessed_image[np.newaxis, :, :, :])
        predicted_class = np.argmax(prediction)
    else:
        prediction = model.predict(preprocessed_image.reshape(1, -1))
        predicted_class = int(prediction)

    return predicted_class

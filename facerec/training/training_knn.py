import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the preprocessed and normalized data
train_data = ImageDataGenerator().flow_from_directory(
    'data_preprocessed_cropped_normalized/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_data = ImageDataGenerator().flow_from_directory(
    'data_preprocessed_cropped_normalized/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Create the feature and label arrays for training
train_features = []
train_labels = []

for i in range(len(train_data)):
    batch_features, batch_labels = train_data[i]
    for j in range(len(batch_features)):
        train_features.append(batch_features[j])
        train_labels.append(np.argmax(batch_labels[j]))

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features.reshape(train_features.shape[0], -1), train_labels)

# Save the trained model
with open('trained_models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

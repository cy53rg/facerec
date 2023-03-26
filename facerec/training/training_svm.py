import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load preprocessed data
X_train, y_train = pickle.load(open('data_preprocessed_cropped_normalized/train/train_data.pkl', 'rb'))
X_val, y_val = pickle.load(open('data_preprocessed_cropped_normalized/train/val_data.pkl', 'rb'))

# Train SVM model
model = svm.SVC(kernel='linear', C=1, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Calculate accuracy on validation set
accuracy = accuracy_score(y_val, y_pred)

# Save model
pickle.dump(model, open('trained_models/svm_model.pkl', 'wb'))

print(f'SVM model trained with {accuracy*100:.2f}% accuracy on validation set.')

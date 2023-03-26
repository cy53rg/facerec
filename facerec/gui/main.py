import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from cnn_model import create_cnn_model
from knn_model import create_knn_model
from svm_model import create_svm_model
import joblib

# rest of the code here

class App:

    def __init__(self, window, window_title):

        self.window = window
        self.window.title(window_title)

        self.video_capture = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.cnn_model = create_cnn_model('training/trained_models/cnn_model.h5')
        self.knn_model = create_knn_model('training/trained_models/knn_model.sav')
        self.svm_model = create_svm_model('training/trained_models/svm_model.sav')

        self.canvas = tk.Canvas(window, width=self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):

        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:

            # Get face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (50, 50))

            # Make predictions using models
            cnn_prediction = self.cnn_model.predict(face_roi.reshape(-1, 50, 50, 1))
            knn_prediction = self.knn_model.predict(face_roi.reshape(1, -1))[0]
            svm_prediction = self.svm_model.predict(face_roi.reshape(1, -1))[0]

            # Draw rectangle around face and label predictions
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'CNN: '+str(cnn_prediction[0]), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, 'KNN: '+str(knn_prediction), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, 'SVM: '+str(svm_prediction), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


if __name__ == '__main__':
    App(tk.Tk(), "Facial Recognition App")

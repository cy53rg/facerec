from sklearn import svm

def get_svm_model():
    model = svm.SVC(kernel='linear', C=1, random_state=42)

    return model

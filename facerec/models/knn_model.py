from sklearn.neighbors import KNeighborsClassifier

def get_knn_model(n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    return model

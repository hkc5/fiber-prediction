import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA

class FiberPredictor():
    def __init__(self):
        self.model = self.build_model()


    def build_model(self):
        scaler= StandardScaler()
        pca_var= PCA(2)
        svr= SVR(kernel= "rbf", C= 100, gamma= .01)

        model= Pipeline([
            ("Scaler", scaler),
            ("PCA", pca_var),
            ("SVR", svr)
        ])
        
        return model

    def train_model(self, X_train, y_train):
        model = self.model
        model.fit(X_train, y_train)
        return model
    

    

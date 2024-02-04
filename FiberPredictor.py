from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA

def FiberPredictor():
    scaler= StandardScaler()
    pca_var= PCA(2)
    svr= SVR(kernel= "rbf", C= 100, gamma= .01)

    model= Pipeline([
        ("Scaler", scaler),
        ("PCA", pca_var),
        ("SVR", svr)
    ])
    
    return model
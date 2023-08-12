import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def get_feature_vector(image_directory, grid_q= 4, orientation= 8, edge= False):
    img= cv2.imread(image_directory, cv2.IMREAD_GRAYSCALE)
    if edge== True:
        img= cv2.Canny(img, 100, 200)

    if type(grid_q)== np.ndarray: # ndarray path
        hog_features= []
        for gq in grid_q:
            hf= get_feature_vector(img_dir, grid_q= gq, orientation= 10, edge= False)
            hog_features= np.append(hog_features, hf)
    else: # integer path
        ppc= (int(img.shape[0]/grid_q), int(img.shape[1]/grid_q))
        hog_features= hog(img, orientations= 12, pixels_per_cell= (int(img.shape[0]/grid_q), int(img.shape[1]/grid_q)),
                        cells_per_block=(1, 1), feature_vector= True)
    return hog_features

# GRIDQ= 4
GRIDQ= np.arange(1, 6)

feature_matrix= []
img_directories= []
file_list= os.listdir("./pictures")
for i in range(1, 1500):
    img_name= str(i) + ".png"
    img_dir= "./pictures/" + str(i) + ".png"
    if img_name in file_list:
        feature_vec= get_feature_vector(img_dir, grid_q= GRIDQ, orientation= 10, edge= False)
        feature_matrix.append(feature_vec)
        img_directories.append(img_dir)

feature_matrix= np.array(feature_matrix)
img_directories= np.array(img_directories)

with open("./pictures/labels.csv") as f:
    l= f.read()
    try: # Mac
        int(l[1:2])
        l= l[1:]
    except: # Windows
        l= l[3:]
    
    labels= np.array(l.split(), dtype= int)

print(f"Total picture count: {img_directories.size}")
print(f"Total label count: {labels.size}")
print(f"Total feature count: {feature_matrix.shape[1]}")

zero_ratio= 1 - np.count_nonzero(feature_matrix)/feature_matrix.size 
print(f"Zero feature ratio: {100*zero_ratio:.1f}%", )

unique_feature, c_unique_feature= np.unique(feature_matrix, return_counts= True)
unique_ratio= 1 - np.sum(np.sort(c_unique_feature[c_unique_feature!= 1]))/feature_matrix.size 
print(f"Unique feature ratio: {100*unique_ratio:.1f}%")
print(25*"-")

def test_model(X, y, img_directories, no_cycles= 10):
    MAE_list= []
    top_error_dict= {}
    for i in range(no_cycles):
        X_train, X_test, y_train, y_test, i_train, i_test= train_test_split(X, y, np.arange(labels.size), test_size= .3)

        scaler= StandardScaler()

        pca= PCA()
        principal_components= pca.fit_transform(X_train)
        var_csum= np.cumsum(pca.explained_variance_ratio_)

        required_variance= .7
        no_components= sum(pca.explained_variance_ratio_.cumsum() < required_variance) + 1
        pca_var= PCA(no_components)

        svr= SVR(C= 100, gamma= .01)

        model= Pipeline([
            ("Scaler", scaler),
            ("PCA", pca_var),
            ("SVR", svr)
        ])

        model.fit(X_train, y_train)
        y_pred= model.predict(X_test)

        mae= mean_absolute_error(y_test, y_pred)
        MAE_list.append(round(mae, 2))
        
        y_error= abs(y_pred-y_test)
        y_error_sorted= np.sort(y_error)[::-1]

        for i in range(3):
            i_max_error= np.where(y_error== y_error_sorted[i])[0]
            i_abs= i_test[i_max_error][0]
            img_i= img_directories[i_abs][11:]
            top_error_dict[img_i] = top_error_dict.get(img_i, 0) + 1

    top_error_dict= {key: value for key, value in sorted(top_error_dict.items(), key= lambda item: item[1], reverse= True)}
    MAE_list= np.array(MAE_list)
    return MAE_list, top_error_dict

MAE_list, top_error_dict= test_model(feature_matrix, labels, img_directories, 100)
print(MAE_list)
print(round(np.mean(MAE_list), 2))
print(top_error_dict)
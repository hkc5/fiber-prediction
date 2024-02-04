import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

GRIDQ_default = np.arange(1,6)
ORIENTATION_default = 10

def get_img(image_directory, rotated= False):
    image = Image.open(image_directory)
    image = image.convert('L')
    return image

def get_feature_vector(image, grid_q, no_orientation=10):
    if type(grid_q)== np.ndarray: # ndarray path
        hog_features= []
        for gq in grid_q:
            hf= get_feature_vector(image, grid_q= gq, no_orientation= no_orientation)
            hog_features= np.append(hog_features, hf)
    else: # integer path
        ppc= (image.size[0]//grid_q, image.size[1]//grid_q)
        hog_features= hog(image, orientations= no_orientation, pixels_per_cell= ppc,
                        cells_per_block=(1, 1), feature_vector= True)
    return hog_features


def augment_data(img, label, img_dir, img_directories, feature_matrix, labels, GRIDQ=GRIDQ_default, ORIENTATION=ORIENTATION_default):
    # Rotations
    for i in range(1, 4):
        a_img = img.rotate(90*i)
        a_feature_vec = get_feature_vector(a_img, grid_q=GRIDQ, no_orientation=ORIENTATION)
        a_label = (label+90*i)%180
        img_directories.append(img_dir)
        feature_matrix.append(a_feature_vec)
        labels.append(a_label)
    # Flips
    a_img = ImageOps.mirror(img)
    a_feature_vec = get_feature_vector(a_img, grid_q=GRIDQ, no_orientation=ORIENTATION)
    a_label = (180-label)%180
    img_directories.append(img_dir)
    feature_matrix.append(a_feature_vec)
    labels.append(a_label)
    
    a_img = ImageOps.flip(img)
    a_feature_vec = get_feature_vector(a_img, grid_q=GRIDQ, no_orientation=ORIENTATION)
    a_label = (180-label)%180
    img_directories.append(img_dir)
    feature_matrix.append(a_feature_vec)
    labels.append(a_label)
    
    a_img = ImageOps.mirror(ImageOps.flip(img))
    a_feature_vec = get_feature_vector(a_img, grid_q=GRIDQ, no_orientation=ORIENTATION)
    a_label = label
    img_directories.append(img_dir)
    feature_matrix.append(a_feature_vec)
    labels.append(a_label)


def get_data(file_dir, csv_dir, i_start=1, augment=False, GRIDQ=GRIDQ_default, ORIENTATION=ORIENTATION_default):
    bio_csv= np.loadtxt(csv_dir)

    feature_matrix= []
    img_directories= []
    labels= []

    for index, label in enumerate(bio_csv):
        if not np.isnan(label):
            img_dir = file_dir + str(index + i_start) + ".png"
            img = get_img(img_dir)

            feature_vec = get_feature_vector(img, grid_q=GRIDQ, no_orientation=ORIENTATION)
            
            img_directories.append(img_dir)
            feature_matrix.append(feature_vec)
            labels.append(label)

            if augment:
                augment_data(img, label, img_dir, img_directories, feature_matrix, labels)

    feature_matrix= np.array(feature_matrix)
    labels= np.array(labels)
    img_directories= np.array(img_directories)
    return feature_matrix, labels, img_directories

def slice_image(image, voxel_size):
    image_width, image_height = image.size
    sliced_list= []
    pos_list= np.empty((0, 2))

    for y in range(0, image_height-voxel_size+1, voxel_size):
        for x in range(0, image_width-voxel_size+1, voxel_size):
            box = (x, y, x + voxel_size, y + voxel_size)
            voxel = image.crop(box)
            sliced_list.append(voxel)
            pos_list= np.append(pos_list, [x, y])

    return sliced_list, pos_list.reshape((-1, 2))
    
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from feature_extraction import *

#========================================== To Submit ======================================
def create_mask_for_test_image(img_path,classifier,patch_size):
    # Will output a mask for the image located at image_size, using classifier.predict 
    img = load_image(img_path)
    img_patches = img_crop(img, patch_size, patch_size)
    
    w = img.shape[0]
    h = img.shape[1]
    
    Xi = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Zi = classifier.predict(Xi)
    predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
    return predicted_im
    

#========================================== To Load ======================================
def get_Xtrain_Ytrain(root_dir,max_number_img,patch_size):
    # Will get the Xtrain and Ytrain from the data using the extract_features_2d

    imgs,gt_imgs = load_train(root_dir,max_number_img)
    assert len(imgs) == len(gt_imgs), 'There are {} gt_imgs and {} imgs => a problem occured'.format(len(gt_imgs),len(imgs))

    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]

    # Linearize list of patches 
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    return X,Y

def load_train(root_dir,max_number_img = 10):
    # Will load all the images and corresponding ground truth
    
    root_dir = root_dir + "training/"
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    print("{} training images are available".format(len(files)))

    n = min(max_number_img, len(files)) # Load maximum 10 images by default
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " corresponding groundtruth")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    return imgs,gt_imgs


def get_Xtest(root_dir,patch_size):
    imgs = load_test(root_dir)

    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    # Linearize those patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def load_test(root_dir):
    #root_dir should be the folder in which you have training and test_set_images
    test_dir = root_dir + "test_set_images/"
    test_folders = os.listdir(test_dir)
    imgs = []
    for i in range(len(test_folders)):
        if test_folders[i].startswith("test_"):
            img_files = os.listdir(test_dir + test_folders[i])
            imgs.append(load_image(test_dir + test_folders[i]+'/'+img_files[0]))
    print("{} test images were loaded".format(len(imgs)))
    return imgs

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


#========================================== Provided ======================================
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Will produce the list of patches (of size w x h obtained from the image)
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
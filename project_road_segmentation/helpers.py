import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import matplotlib.image as mpimg
import re
import scipy.misc
from PIL import Image
from feature_extraction import *

#========================================== To Submit ======================================
def create_mask_pred_all_test(submission_path,test_dir_path,classifier,patch_size):
    """This method will use classifier to predict the mask of each image, 
    save the mask in the respective test folders
    and finally output the submission file in submission path
    """
    masks_file_names = []
    for i in range(1,51):
        suffix = 'test_{}/test_{}.png'.format(i,i)
        img_path = test_dir_path + suffix
        # We compute the mask
        predicted_im = create_mask_for_test_image(img_path,classifier,patch_size)
        suffix_pred = 'test_{}/test_{}_mask.png'.format(i,i)
        pred_path = test_dir_path + suffix_pred
        masks_file_names.append(pred_path)
        # We save the mask
        scipy.misc.imsave(pred_path, predicted_im)
    masks_to_submission(submission_path,masks_file_names)


def create_mask_for_test_image(img_path,classifier,patch_size):
    """output a mask for the image located at image_path, using classifier.predict"""
    img = load_image(img_path)
    img_patches = img_crop(img, patch_size, patch_size)
    
    w = img.shape[0]
    h = img.shape[1]
    
    Xi = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Zi = classifier.predict(Xi)
    predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
    return predicted_im

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    # we get the very last element of the path, to get the filename
    filename = re.split('\/+', image_filename)[-1]
    img_number = int(re.search(r"\d+", filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(image_filenames)):
            fn = image_filenames[i]
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
    

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

def balance_training_set(X_train,Y_train):
    """Will balance the training set so that we train on the same amount of sample for each class"""
    print ('Balancing training data...')
    idx0 = [i for i, j in enumerate(Y_train) if j == 0]
    idx1 = [i for i, j in enumerate(Y_train) if j == 1]
    c0_old = len(idx0)
    c1_old = len(idx1)
    min_c = min(c0_old, c1_old)

    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    Y_train = Y_train[new_indices]
    X_train = X_train[new_indices,:]
    assert Y_train.shape[0] == X_train.shape[0]

    Y0 = [i for i, j in enumerate(Y_train) if j == 0]
    Y1 = [i for i, j in enumerate(Y_train) if j == 1]
    c0_new = len(Y0)
    c1_new = len(Y1)
    print("Old (c0,c1) = ({},{})\nNew (c0,c1) = ({},{})".format(c0_old,c1_old,c0_new,c1_new))
    return X_train,Y_train

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
    imgs = []
    for i in range(1,51):
        imgs.append(load_image(test_dir + 'test_'+str(i)+'/'+'test_'+str(i)+'.png'))
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
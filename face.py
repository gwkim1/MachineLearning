from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re
import math
import time
from PIL import Image, ImageDraw

def get_grey(filename):
    """
    Load the image, return the np.array of greyscale-converted image
    """
    image = misc.imread(filename)
    grey = np.zeros((image.shape[0], image.shape[1]))
    for i in range(len(image)):
        for j in range(len(image[i])):
          grey[i][j] = np.average(image[i][j])
    return grey

def get_iimages(path1, path2, savefile):
    """
    Store the greyscale-converted image in a N_images * (64*64) matrix.
    The matrix contains integral image representations
    Two paths needed for faces and non_faces directories
    """
    files1 = [f for f in listdir(path1) if isfile(join(path1, f)) and f[0] == 'f']
    files2 = [f for f in listdir(path2) if isfile(join(path2, f)) and f[0] != '.']
    
    img = get_grey(path1 + "\\" + files1[0])
    iimages1 = np.array([[0 for col in range(img.shape[0] * img.shape[1])] for row in range(len(files1))], dtype="float")
    iimages2 = np.array([[0 for col in range(img.shape[0] * img.shape[1])] for row in range(len(files2))], dtype="float")
    
    cnt = 0
    
    for filename in files1:
        print cnt
        imgnum = int(re.findall('[0-9]+', filename)[0])
        img = get_grey(path1 + "\\" + filename)

        for row in range(img.shape[0]):
            rowsum = 0
            for col in range(img.shape[1]):
                index = row*img.shape[1] + col
                
                
                iimages1[imgnum][index] = rowsum + img[row][col]
                if row > 0:
                    iimages1[imgnum][index] += iimages1[imgnum][index-img.shape[1]]
                rowsum += img[row][col]
        cnt += 1

    for filename in files2:
        print cnt
        imgnum = int(re.findall('[0-9]+', filename)[0])
        img = get_grey(path2 + "\\" + filename)

        for row in range(img.shape[0]):
            rowsum = 0
            for col in range(img.shape[1]):
                index = row*img.shape[1] + col
                iimages2[imgnum][index] = rowsum + img[row][col]
                if row > 0:
                    iimages2[imgnum][index] += iimages2[imgnum][index-img.shape[1]]
                rowsum += img[row][col]
        cnt += 1
    
    final_iimages = np.vstack((iimages1, iimages2))
    np.save(savefile, final_iimages)


def get_iimages_for_class(img):
    """
    Get the integral image representation for the testing image
    """
    iimages = np.array([[0 for col in range(img.shape[1])] for row in range(img.shape[0])], dtype="float")

    for row in range(img.shape[0]):
        print row
        rowsum = 0
        for col in range(img.shape[1]):
            iimages[row][col] = rowsum + img[row][col]
            
            if row > 0:
                iimages[row][col] += iimages[row-1][col]
            rowsum += img[row][col]
    
    return iimages

def get_patches_for_class(classimg, stride, patchsize, ymin, ymax):
    """
    Get the list of patchsize*patchsize 1-dim patch arrays and list of coordinates
    That would be used for testing the trained model.
    
    Could set ymin-ymax range because this function takes a long time.
    """
    startx, starty = 0, ymin
    
    patches = []
    coordinates = []
    
    prevy = 0
    while startx <= (classimg.shape[1] - patchsize) and starty <= (classimg.shape[0] - patchsize) and ymin <= starty and starty <= ymax:
        if prevy != starty:
            print "starty:", starty
        patch = []
        for row in range(patchsize):
            for col in range(patchsize):
                patch.append(classimg[starty+row][startx+col])
        
        patches.append(patch)
        coordinates.append((startx, starty))
        
        prevy = starty
    
        if startx + stride <= (classimg.shape[1] - patchsize):
            startx += stride
        else:
            startx = 0
            starty += stride
            
    return np.array(patches), coordinates    
    
def get_featuretbl(stride, minsmall, maxsmall, minlarge, maxlarge, div):
    """
    Get the N * 8 np array, which each row being a feature
    
    The smaller length and the larger length of each black-white rectangle
    is limited by min-max values.
    
    div only counts lengths that are divisible by div (to reduce num of features)
    """
    featuretbl = []
    # Create upper and lower rectangle feature
    # Any combination of min-max * min-max would be possible
    for i in range(maxlarge - minlarge + 1):
        for j in range(maxsmall - minsmall + 1):
            if i % div == 0 and j % div == 0:
                width = minlarge + i
                height = minsmall + j
                print width, height
                
                startx = 0
                starty = 0
                # while the width does not exceed the 64*64 grid
                while (startx + width <= 64 and starty + 2*height <= 64):
                    newrow = [startx, starty, startx + width, starty + height,
                              startx, starty+height, startx + width, starty + 2*height]
                    featuretbl.append(newrow)
                    
                    # check whether adding a stride is safe, and then add
                    if startx + stride + width <= 64:
                        startx += stride
                    # only move towards bottom when we can't move to the right
                    # set startx back to 0
                    else:
                        startx = 0
                        starty += stride
    
    # Now, do the same for the left and right rectangles
    for i in range(maxsmall - minsmall + 1):
        for j in range(maxlarge - minlarge + 1):
            if i % div == 0 and j % div == 0:
                width = minsmall + i
                height = minlarge + j
                print width, height
                
                startx = 0
                starty = 0
                while (startx + 2*width <= 64 and starty + height <= 64):
                    newrow = [startx, starty, startx + width, starty + height,
                              startx+width, starty, startx + 2*width, starty + height]
                    featuretbl.append(newrow)

                    if startx + stride + 2*width <= 64:
                        startx += stride
                    else:
                        startx = 0
                        starty += stride    

    return np.array(featuretbl)        
  
    
def compute_feature(i, f, iimages, ftbl):
    """
    return the value of feature number f in image number i (using integral image)
    i: example num. 64*64 array
    f: number indicating rownum in featuretbl
    """
    row1, col1 = ftbl[f][0], ftbl[f][1]    
    row1p, col1p = ftbl[f][2], ftbl[f][3]
    row2, col2 = ftbl[f][4], ftbl[f][5]
    row2p, col2p = ftbl[f][6], ftbl[f][7]

    if row1 == 0 and col1 != 0:
        black = iimages[i][(row1p-1)*64+(col1p-1)] - iimages[i][(row1p-1)*64+(col1-1)]  
    elif row1 != 0 and col1 == 0:
        black = iimages[i][(row1p-1)*64+(col1p-1)] - iimages[i][(row1-1)*64+(col1p-1)]   
    elif row1 == 0 and col1 == 0:
        black = 0 + iimages[i][(row1p-1)*64+(col1p-1)]
    else:
        black = iimages[i][(row1-1)*64+(col1-1)] + iimages[i][(row1p-1)*64+(col1p-1)] - \
                iimages[i][(row1-1)*64+(col1p-1)] - iimages[i][(row1p-1)*64+(col1-1)]        
    
    if row2 == 0 and col2 != 0:
        white = iimages[i][(row2p-1)*64+(col2p-1)] - iimages[i][(row2p-1)*64+(col2-1)]  
    elif row2 != 0 and col2 == 0:
        white = iimages[i][(row2p-1)*64+(col2p-1)] - iimages[i][(row2-1)*64+(col2p-1)]   
    elif row2 == 0 and col2 == 0:
        white = 0 + iimages[i][(row2p-1)*64+(col2p-1)]
    else:
        white = iimages[i][(row2-1)*64+(col2-1)] + iimages[i][(row2p-1)*64+(col2p-1)] - \
                iimages[i][(row2-1)*64+(col2p-1)] - iimages[i][(row2p-1)*64+(col2-1)]            

    return black - white
    
 
def find_p_theta(examples, labels, f, ftbl, weights):
    """
    Find the weak learner for a specific feature f at a certain weight
    
    examples would be N * 64^2 array, each row being an example image
    labels would be an array [1, 1, 1, ... -1, -1, -1] for face/nonface
    
    f: feature num
    """    
    f_values = []

    for j in range(len(examples)):
        f_values.append([compute_feature(j, f, examples, ftbl), j, labels[j], weights[j]])    
    f_values.sort()

    # for first iteration
    left_term = 0
    min_i = 0
    
    if f_values[0][2] == 1:
        left_term += f_values[0][3]
    
    for k in range(1, len(f_values)):

        if f_values[k][2] == -1:
            left_term += f_values[k][3]   

    min_error_j = min(left_term, 1-left_term)
    min_left = left_term
    
    # To save runtime, update left_term based on the previous value,
    # instead of always calculating left_term from scratch
    for i in range(1, len(f_values)):
        if f_values[i][2] == 1:
            left_term += f_values[i][3]
        else:
            left_term -= f_values[i][3]
                
        error_j = min(left_term, 1-left_term)
        
        if error_j < min_error_j:
            min_error_j = error_j
            min_i = i
            min_left = left_term
    
    if min_left < 0.5:
        p = 1
    else:
        p = -1
     
    if min_i != len(f_values) - 1:
        theta = (f_values[min_i][0] + f_values[min_i+1][0]) / float(2)
    else:
        theta = f_values[min_i][0]

    return p, theta, min_error_j    
    
def best_learner(examples, labels, ftbl, weights):
    """
    Gets the feature that does the best for a certain adaboost round
    """
    best_f = -1
    min_error = 9999999999
    best_theta = -99999999
    best_p = 0
    
    # Check for each feature. For each feature, find the best p, theta
    for f in range(len(ftbl)):
        if f % 100 == 0:
            print "\ncheck for feature", f
        
        # for each feature, train a classifier. 
        p, theta, min_error_j = find_p_theta(examples, labels, f, ftbl, weights)

        if min_error_j < min_error:
           
            min_error = min_error_j
            best_theta = theta
            best_f = f
            best_p = p
            
    return best_f, best_p, best_theta
   
def get_error(h_t, weights, examples, labels, f, p, theta):
    """
    Check how well the weak learner does on the given examples
    """
    error = 0
    
    for i in range(len(examples)):
        if h_t(examples, i, (f, p, theta)) != labels[i]:
            error += weights[i]
    return error
    
def set_initial_weights(examples, labels):
    """
    Set initial weights so that the total weight of faces and non-faces
    Are always equal (even after excluding false-positive non-faces)
    """
    for i in range(len(labels)):
        if labels[i] == -1:
            first_minus = i
            break

    face_weight = float(0.5) / first_minus
    background_weight = float(0.5) / (len(examples) - first_minus)
    return [float(0.5) / first_minus] * first_minus +  [float(0.5) / (len(examples) - first_minus)] * (len(examples) - first_minus) 
    
def adaboost_new(examples, labels, ftbl, max_fpr):
    """
    Conduct one cascade-cycle of Adaboost
    """
    weights = set_initial_weights(examples, labels)
    
    hs = []
    hargs = []
    alphas = []

    t = 0
    
    while True:
        print "\n iteration", t 
        t += 1
        
        # Find the best weak learner
        f, p, theta = best_learner(examples, labels, ftbl, weights)
        print "f, p, theta:", f, p, theta
        
        # Define h_t based on the acquired parameters, add the function and parameters to lists
        h_t = lambda patches, i, (f, p, theta): 1 if \
            p*(compute_feature(i, f, patches, ftbl) - theta) >= 0 else -1
        hs.append(h_t)
        hargs.append((f, p, theta))
        
        error = get_error(h_t, weights, examples, labels, f, p, theta)

        alpha_t = 0.5 * math.log((1-error)/float(error))
        alphas.append(alpha_t)
        Z_t = 2 * math.pow(  error*(1-error) , 0.5 )        
        print "error, a_t, Z_t:", error, alpha_t, Z_t
        
        for i in range(len(examples)):
            y_i = labels[i]
            x_i = examples[i]
            # Update new weight
            weights[i] = weights[i] * math.exp( -alpha_t * y_i * h_t(examples, i, (f, p, theta)) ) / (Z_t)
        
        # Get fpr and only pick the positively classified examples and labels
        fpr, pos_examples, pos_labels = test_fpr(hs, hargs, alphas, examples, labels)
        print "fpr:", fpr
        
        # Stop iteration only when the false positive rate gets below target
        if fpr <= max_fpr:
            return hs, hargs, alphas, pos_examples, pos_labels

    return hs, hargs, alphas, pos_examples, pos_labels    
    

def test_fpr(hs, hargs, alphas, examples, labels):
    """
    After each adaboost iteration, returns the fpr
    + return a new example set with true positives + false positives
    """
    final_theta, real_ada = real_adaboost(hs, hargs, alphas, examples, labels)
    
    pos_examples = []
    pos_labels = []
    
    fp = 0
    neg_count = 0
    for i in range(len(examples)):
        result = real_ada(hs, hargs, alphas, examples, i)
        
        if labels[i] == -1:
            neg_count += 1
        
        if result == 1 and labels[i] == -1:
            fp += 1
        if result == 1:
            pos_examples.append(examples[i])
            pos_labels.append(labels[i])
    
    return fp / float(neg_count), pos_examples, pos_labels
    

def cascade_adaboost(examples, labels, ftbl, max_fpr, max_final_fpr):
    """
    Conduct a full Adaboost with multiple cascades
    """
    initial_neg_count = 0
    train_examples = []
    train_labels = []
    
    h_arg_alpha_ada_list = []
    
    for i in range(len(examples)):
        if labels[i] == -1:
            initial_neg_count += 1
        
        train_examples.append(examples[i])
        train_labels.append(labels[i])
    
    initial_pos_count = len(examples) - initial_neg_count
    
    cascade = 0
    
    while True:
        print "\n\ncascade", cascade
        hs, hargs, alphas, train_examples, train_labels = adaboost_new(train_examples, train_labels, ftbl, max_fpr)
        
        final_theta, real_ada = real_adaboost(hs, hargs, alphas, examples, labels)
        cascade_tuple = (hs, hargs, alphas, real_ada, final_theta)
        h_arg_alpha_ada_list.append(cascade_tuple)
        
        cascade_fpr = (len(train_examples) - initial_pos_count) / float(initial_neg_count)
        print "cascade fpr:", cascade_fpr
        if cascade_fpr <= max_final_fpr:            
            return h_arg_alpha_ada_list
        
        cascade += 1
    
    return h_arg_alpha_ada_list
   
    
def f_from_hs(hs, hargs, alphas, patches, i):
    """
    Calculates the sum of all alpha_t * h_t(x) for a certain example
    (over all Adaboost iteraion)
    """
    sum = 0
    for t in range(len(hs)):
        sum += alphas[t] * hs[t](patches, i, hargs[t])
    
    return sum
    
def real_adaboost(hs, hargs, alphas, examples, labels):
    """
    Unlike Vanilla adaboost, set a threshold that ensures no false positives
     
    Eventually, this would return a function for the final hypothesis
    """
    min_val = 99999999999
    
    for i in range(len(examples)):
        # Only check the minimum for examples that have label 1
        if labels[i] == 1:
            # f_from_hs would pick a single example i, and then sum update
            # all alphat ht (x) over all T
            sum_of_ah = f_from_hs(hs, hargs, alphas, examples, i)
            if sum_of_ah < min_val:
                min_val = sum_of_ah

    final_theta = min_val
    
    return final_theta, lambda hs, hargs, alphas, examples, i: 1 if f_from_hs(hs, hargs, alphas, examples, i) - final_theta >= 0 else -1

    
def reconstruct_haaat_list(hargs_list, alphas_list, ftbl):
    """
    Reconstruct a list of [hs, hargs, alphas] from saved lists
    Needed because function list cannot be saved to a file
    """
    hs = []
    for cascade in range(len(hargs_list)):
        iter_hs = []
        
        for iter in range(len(hargs_list[cascade])):
            (f, p, theta) = hargs_list[cascade][iter]

            h_t = lambda patches, i, (f, p, theta): 1 if \
                p*(compute_feature(i, f, patches, ftbl) - theta) >= 0 else -1
            
            iter_hs.append(h_t)
                
        hs.append(iter_hs)
    
    haaat_list = []

    for cascade in range(len(hargs_list)):
        haaat_list.append([hs[cascade], hargs_list[cascade], alphas_list[cascade]])

    return haaat_list
 
def classify_0(haaat_list, examples, i):
    """
    Based on the haaat_list that contains information about trained model,
    Determine whether the ith example is a face or not
    """
    for tuple in haaat_list:
        hs = tuple[0]
        hargs = tuple[1]
        alphas = tuple[2]
        # During the testing stage, final_theta is set to 0 to improve performance
        real_ada_0 = lambda hs, hargs, alphas, examples, i: 1 if f_from_hs(hs, hargs, alphas, examples, i) >= 0 else -1
        
        result = real_ada_0(hs, hargs, alphas, examples, i)
        if result == -1:
            return -1
    return 1

def get_faces_0(haaat_list, patches, coordinates):
    """
    With the given list of testing images and coordinates,
    Return the coordinates corresponding to face patches
    """
    faces = []
    
    for i in range(len(patches)):
        if classify_0(haaat_list, patches, i) == 1:
            faces.append(list(coordinates[i]))
    return faces    

    
"""
Functions to manually draw and inspect each patch identified as a face
"""    
def check_each_face(grey, face_coord, i):
    print face_coord[i]
    plt.imshow(grey[face_coord[i][1]:face_coord[i][1]+64, face_coord[i][0]:face_coord[i][0]+64], cmap='gray')

def check_each_face_coord(grey, coord):
    print coord
    plt.imshow(grey[coord[1]:coord[1]+64, coord[0]:coord[0]+64], cmap='gray')

def pick_one_face_and_filter(face_coord, gap, filter_threshold):
    """
    Apply the exclusion rule to avoid overlapping patches all identifying a single face    
    Apply filter_threshold to exclude false positives which only have a few positively classified patches in its proximity
    """
    pdic = pick_one_face(face_coord, gap)
    cdic = get_coords(face_coord, gap)
    
    face = []
    nonface = []

    for i in range(len(pdic)):
        if len(cdic[i]) >= filter_threshold:
            face.append((pdic[i], i))
        else:
            nonface.append((pdic[i], i))
            
    return face, nonface, cdic

def get_coords(face_coord, gap):    
    cnt = 0
    patch_dict = {}
    coord_dict = {}
    avg_dict = {}

    # Assign faces into clusters
    for i in range(len(face_coord)):
        face = face_coord[i]
        coord_dict, cnt = assign_face(patch_dict, avg_dict, gap, face, cnt)
    return coord_dict
    
    
def assign_face(patch_dict, avg_dict, gap, face, cnt):
    """
    Assign each positively classified patch to a group of adjacent patches
    """
    #print "\nassign face:", face
    
    assigned = False
    if len(patch_dict) == 0:
        patch_dict[0] = [list(face)]
        avg_dict[0] = list(face)
    else:
        for cnt, facelist in patch_dict.items():      
            if close_to_average(cnt, face, facelist, avg_dict, gap):
                facelist.append(list(face))
                avg_dict[cnt] =  [ (avg_dict[cnt][0] * (len(facelist)-1) + face[0]) / float(len(facelist)) , (avg_dict[cnt][1] * (len(facelist)-1) + face[1]) / float(len(facelist))  ]
                assigned = True
                break
        
        if not assigned:
            cnt += 1
            patch_dict[cnt] = [list(face)]
            avg_dict[cnt] = list(face)
    
    return patch_dict, cnt
 
def close_to_average(cnt, face, facelist, avg_dict, gap):
    """
    Determine the proximity of a patch to the average location of a patch group
    """
    if math.sqrt((avg_dict[cnt][0] - face[0])**2 + (avg_dict[cnt][1] - face[1])**2) <= gap: 
        return True
    else:
        return False

def common_list_all(biglist):
    """
    Filter the patches that exist in all lists
    """
    common = common_list(biglist[0], biglist[1])
    for i in range(2, len(biglist)):
        common = common_list(common, biglist[i])
    return common


def draw_result(coordinate_list, filename):
    """
    Draw rectangles around all patches classified as faces (after exclusion rule)
    """
    im = Image.open("class.jpg")

    draw = ImageDraw.Draw(im)
    
    for coord in coordinate_list:
        draw.rectangle((coord[0][0], coord[0][1], coord[0][0]+63, coord[0][1]+63), fill=None, outline=250)

    im.save(filename)  
        

print "success"    







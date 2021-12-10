from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
import shutil

def load_one_id(path, n):
    ans = [] 
    if (os.path.isdir(path)):
        k = 1
        img_path = path + 'ID' + str(n) + '_' + str(k) + '.jpg'
        while(os.path.isfile(img_path)):
            img = np.array(image.imread(img_path))
            height, width = img.shape
            img = img.reshape(1, height*width)
            img = img[0]
            ans.append(img)
            k = k + 1
            img_path = path + 'ID' + str(n) + '_' + str(k) + '.jpg'
    ans = np.array(ans)
    if (len(ans) == 0):
        return ans
    return np.mean(ans, axis=0)


#Load data image 
data = []
n = 1
while True:
    path = 'image/ORL/p' + str(n) + '/'
    data_ele = load_one_id(path, n)
    if (len(data_ele) == 0):
        break
    data.append(data_ele)
    n = n + 1 
data = np.array(data)
#print(len(data))

#Load resolution of image
img1 = image.imread('1_001.jpg')
data1 = np.array(img1)
height, width = data1.shape

#Load infomation
file = open('info.txt','r')
info = []
while True:
    line = file.readline()
    if not line:
        break
    line_add = line.rstrip('\n').split(',')
    info.append(line_add)
    
file.close()


#Handle PCA
data = data.T
pca = PCA(n_components=0.95)
pca.fit(data.T)

U = pca.components_.T
eig_val = pca.explained_variance_

def dist(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum(np.power((p-q), 2)))

def predict(path):
    test_img = np.array(image.imread(path))
    u = test_img.reshape(height*width, 1) - pca.mean_.reshape(height*width, 1)
    x = U.T.dot(u)

    minclass = -2
    mindist = float('inf')
    for i in range(len(data.T)):
        v = data[:, i].reshape(height*width, 1) - pca.mean_.reshape(height*width, 1)
        v = U.T.dot(v)
        distance = dist(x, v)
        if distance < mindist:
            minclass = i
            mindist = distance
    #check again
    path_min = 'image/ORL/p' + str(minclass+1) + '/' + 'ID'+ str(minclass+1) + '_1' + '.jpg'
    img_min = np.array(image.imread(path_min)).reshape(height*width, 1)
    w_min = img_min - pca.mean_.reshape(height*width, 1)
    w_min = U.T.dot(w_min)

    w_predict = data[:, minclass].reshape(height*width, 1) - pca.mean_.reshape(height*width,1)
    w_predict = U.T.dot(w_predict)
    r_distance = dist(w_min, w_predict)
    #print(minclass)
    #print(r_distance)
    #print(mindist)
    if(mindist<=r_distance or mindist<r_distance+0.8*r_distance):
        return minclass, path
    else:
        announce = 'Not found'
        return announce

def print_info(predict):
    if(predict == 'Not found'):
        print('Warning: Denied')
        return
    minclass, src = predict
    print('----------------------------------------------')
    print('VIET NAM NATIONAL UNIVERSITY, HO CHI MINH CITY')
    print('            UNIVERSITY OF SCIENCE             ')
    print('               STUDENT ID CARD                ')
    
    print('Name:\t' + info[minclass][0])
    print('Date of birth:\t' + info[minclass][1])
    print('Class ID:\t' + info[minclass][2])
    print('Student ID:\t' + info[minclass][3])
    print('----------------------------------------------')
    choose = input('Do you want an image of student extracted for next processing? (Y/N)      ')
    if choose == 'Y':
        i = 1
        path = 'image/ORL/p' + str(minclass+1) + '/' + 'ID' + str(minclass+1) + '_' + str(i)+ '.jpg'
        while(os.path.isfile(path)):
            i = i+1
            path = 'image/ORL/p' + str(minclass+1) + '/' + 'ID' + str(minclass+1) + '_' + str(i)+ '.jpg'
        shutil.copyfile(src, path)
        print("Done")
    if choose == 'N':
        return
    if choose != 'N' and choose != 'Y':
        print('You entered the false syntax, the programing is ending!')
        
    

def register(path_img):
    if predict(path_img) == 'Not found':
        print('----------------------------------------------')
        print('           REGISTER NEW STUDENT\n\n')
        print('Please complete the form')
        name = input('Name: ')
        birth = input('Date of birth: ')
        classid = input('Class ID: ')
        studentid = input('Student ID: ')
        
        file = open('info.txt', 'a')
        line = name + ',' + birth + ',' + classid + ',' + studentid
        file.write('\n')
        file.write(line)
        file.close()
        
        i = 1
        path = 'image/ORL/p' + str(i) + '/'
        while(os.path.isdir(path)):
            i = i+1
            path = 'image/ORL/p' + str(i) + '/'
        os.mkdir(path)
        path = path+'ID'+str(i)+'_1.jpg'
        shutil.copyfile(path_img, path)
        print("Done")
    else:
        print('This student is already registered')
while True:
    print('----------------------------------------------')
    print('VIET NAM NATIONAL UNIVERSITY, HO CHI MINH CITY')
    print('            UNIVERSITY OF SCIENCE             ')
    print('1:\tCard scan (SC)')
    print('2:\tRegister new student (R)')
    print('3:\tQuit (Q)')
    opt = input('Enter your option:\t')
    if (opt == "SC"):
        path = input('Enter you image\'s path: \t')
        print_info(predict(path))
    elif (opt == "R"):
        path = input('Enter you image\'s path: \t')
        register(path)
    elif (opt == "Q"):
        print("Quitting..........")
        break
    else:
        print("You enter the false syntax")
 




    



        





    
    







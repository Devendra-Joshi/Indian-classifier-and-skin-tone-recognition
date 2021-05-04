import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix

pred_number = 0
name = ''


#def rename(path):
    
#    rename_i = 0
#    for filename in os.listdir(path):
       
#        New_img =str(rename_i) + ".jpg"
#        Img_source =path + filename
#        New_img =path + New_img
#        os.rename(Img_source, New_img)
#        rename_i += 1

def crop_images(path):
    
    label = 0
    name = str(input('\n\tEnter the name of image: '))
    path_img = path + '/'+name
    print(path_img)
    img = cv2.imread(path_img)
    output = 1
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(r'C:\Users\thete\Desktop\ML-major project\haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors = 15, minSize=(100, 100))
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        faces = img[y:y + h, x:x + w]
        
        if len(faces)>=1:
            output = 0
            new_path = 'cropped/'+str(label)+'.jpg'
            cv2.imwrite(new_path, faces)
            print('\nHuman')

    return output

def train_model():

    target = []     #The output here
    images = []     #datan

    flat_data = []  #flatten data

    datadir = input('Enter the path of your dataset(image dataset): ')

    Category_no = int(input('Enter the number of folder inside the dataset? '))
    categories = []
    for number in range(Category_no):

        categories.append(input("Enter name of folders inside the dataset: "))

    for category in categories:
        num = categories.index(category)        #Label Encoding values
        path = datadir + '/' + category

        onlyfiles = next(os.walk(path))[2]
        print(len(onlyfiles))
        #rename(path)
        for i in range(len(onlyfiles)):

            img_path = path + '/' + str(i)+'.jpg'           #if getting error that imsg not founf then run the rename function
            img_array = imread(img_path)
            
            img_resized = resize(img_array,(40,40,3))       #resizing of data and also normalizes th value from 0-1
            flat_data.append(img_resized.flatten())         #flattening of data
            images.append(img_resized)
            target.append(num)

    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)


    x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size = 0.2, random_state = 0)

    param_grid = [
                        {'C': [1, 10, 100, 1000], 'kernel':['linear']},
                        {'C': [1, 10, 100, 1000], 'gamma':[0.001,0.0001], 'kernel':['rbf']}
                 ]


    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, param_grid)
    clf.fit(x_train,y_train)


    y_pred = clf.predict(x_test)

    score = accuracy_score(y_pred,y_test)
    print(score)

    pickle.dump(clf,open('img_model.p', 'wb'))

def addon(path, crop,output):

    if output == 0:
        flat_data = []
        print("\n\n\t\t\tTesting image with the predicted model\n")
        model = pickle.load(open('img_model.p','rb'))
        

        for times in range(2):

            if crop == 'y' or crop =='Y':
                name = str(input('\n\tEnter the name of image: '))
                img_path = path + '/'+name

            else:
                img_path = 'cropped/0.jpg'

            img = imread(img_path)
            img_resized = resize(img,(40,40,3))
            if times == 0:
                flat_data.append(img_resized.flatten())
            elif times > 0:
                if targets == 0:
                    print('\n\t\t\tIndian')

                elif targets == 1:
                    print('\n\t\t\tNon Indian')

            if times == 0:
                flat_data = np.array(flat_data)
                targets = model.predict(flat_data)
        return targets



def skin_tone(path,crop,output):

    if output == 0:
        if crop == 'y' or crop =='Y':
            img_path = path + '/'+name
        else:
            img_path = 'cropped/0.jpg'
        img = cv2.imread(img_path)
        print(img_path)

        skin_data = img.mean()
        print(skin_data)

        if skin_data< 94:
            print('\n\n\t\t\tDARK')

        elif skin_data>94 and skin_data< 112:
            print('\n\n\t\t\tMILD')

        elif skin_data>112:
            print('\n\n\t\t\tFAIR')
  

def main():

    choice2 = str(input("\n\t\t\tAre you training the model for first time?(y/n) "))
    if choice2 == 'y' or choice2 == 'Y':
        train_model()
    elif choice2 =='n' or choice2 =='N':
        path = str(input('\n\tPath of the folder for testing :(with forward slash"/") '))
        crop = str(input('\n\tAre the images already cropped(only face?(y/n)) '))
        print('\n\n\t\t\tFirst Phase: Human and Non-Human\n\n')

        if crop == 'n' or crop =='N':
            output = crop_images(path)
            print(output)
        if output == 0:
            print('\n\n\t\t\tSecoond Phase: Indian and Non-Indian\n\n')

            
            if choice2 =='n' or choice2 =='N':
                target = addon(path,crop,output)            

            if target == 0:
                skin_tone(path,crop,output)
            else:
                print('\n\tNon Indian face detected so cannot move further')
        else:
            print("No Human Detected")

if __name__ == '__main__':
   main()

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:48:05 2023

@author: Alfonso Blanco
"""

# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

dirname="Training"
dirnameTest="Test"

######################################################################


import numpy as np

import cv2


import os
import re

import imutils

#####################################################################

def loadCodFilterTraining(dirname):
    thresoldpath = dirname 
    
    
    arry=[]
   
   
    print("Reading codfilters from ",thresoldpath)
        
    Conta=0
    
    for root, dirnames, filenames in os.walk(thresoldpath):
        
       
        for filename in filenames:
           
            
            if re.search("\.(txt)$", filename):
                Conta=Conta+1
                #arry=[]
               
                filepath = os.path.join(root, filename)
              
              
                f=open(filepath,"r")
               
               
                for linea in f:
                    
                    
                    
                    arry.append(int(linea))
                        
              
                f.close() 
               
                  
                
    
    
   
    Y_train=np.array(arry)
    
    return  Y_train
   


#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname 
    
    images = []
    imagesFlat=[]
    Licenses=[]
    arr=[]
    Conta=0
    ContFirst=0
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    
    for root, dirnames, filenames in os.walk(imgpath):
        
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                Conta=Conta+1
                
                filepath = os.path.join(root, filename)
                License=filename[:len(filename)-4]
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (416,416), interpolation = cv2.INTER_AREA) 
                
                images.append(image)
                imagesFlat.append(gray.flatten())
                Licenses.append(License)
                         
    return imagesFlat, Licenses


###########################################################
# MAIN
##########################################################
Y_train=loadCodFilterTraining(dirname)
print(Y_train)
X_train, Licenses=loadimages(dirname)
X_test, LicensesTest=loadimages(dirnameTest)


print("Number of imagenes to test : " + str(len(X_train)))
print("Number of  CodFilters  : " + str(len(Y_train)))

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier
"""
# https://pub.towardsai.net/an-offbeat-approach-to-brain-tumor-classification-using-computer-vision-19c9e7b84664
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# MODEL INSTANTIATION
model = SVC(kernel = 'rbf')
parameters = {'C':[0.1,1,10,100,1000,10000,100000]}
grid_search = GridSearchCV(param_grid = parameters, estimator = model, verbose = 3)
# MODEL TRAINING AND GRID-SEARCH TUNING
grid_search = grid_search.fit(X_train,Y_train)
print(grid_search.best_params_)

model = SVC(kernel = 'rbf', gamma=1/11.0, C=0.1)
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
model =  OneVsRestClassifier(SVC(kernel='linear', probability=True,  max_iter=1000)) #Creates model instance here 
#model =  OneVsRestClassifier(SVC())
#model =  OneVsRestClassifier(SVC(kernel='linear',   max_iter=2000)) #Creates model instance here 
#model =  OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, max_iter=1000)) #Creates model instance here
#model =  OneVsRestClassifier(SVC(kernel='poly', degree=12)) #Creates model instance here
#model =  OneVsRestClassifier(SVC(kernel='rbf',gamma =1/11.0) )
#model =  OneVsRestClassifier(SVC(kernel='sigmoid')) 

Y_train=Y_train.astype(int)
X_train=np.array(X_train)
X_train=X_train.astype(int)

model.fit(X_train, Y_train) #fits model with training data


pickle.dump(model, open("./model.pickle", 'wb')) #save model as a pickled file

model2= pickle.load( open("./model.pickle", 'rb'))
predictions=model2.predict(X_test)
#
TotHits=0
TotFailures=0

NumberImageOrder=0

for i in range (len( LicensesTest)):
    
    # Blur the ROI of the detected licence plate
    # pesimos resultados
    #gray1 = cv2.GaussianBlur(imagesLicenseTest[i] ,    (35,35),0)
    
    
    #cv2.imshow("Prueba", gray1)
    #cv2.waitKey()
   
    
    NumberImageOrder=NumberImageOrder+1
    CodFilter=predictions[i] 
    print(LicensesTest[i] + "  CodFilter = "+ str(CodFilter))
    """
    ret, gray1=cv2.threshold( imagesLicenseTest[i],threshold,255,  cv2.THRESH_BINARY)
    
    
    
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3') 
    text = ''.join(char for char in text if char.isalnum())
    LicenseTest=LicensesTest[i]
    #https://stackoverflow.com/questions/67857988/removing-newline-n-from-tesseract-return-values
    #print(text)
    if (text[0:len(LicenseTest)]==LicenseTest):
       print ("HIT the license is detected as " + text[0:len(LicenseTest)])
       TotHits=TotHits+1
    else:                                             
        # se admite que pueda exisir al principio una posicion sin informacion 
        if text[1:len(LicenseTest)+1]==LicenseTest :
            print ("HIT the license is detected as " + text[1:len(LicenseTest)+1])
            TotHits=TotHits+1
        else:
              print ("Error is detected " + text + " insted the true license  " + LicenseTest)
              TotFailures=TotFailures +1 
print("")   
print(" Total Hits = " + str(TotHits)) 
print(" Total failures = " + str(TotFailures)) 
"""
    
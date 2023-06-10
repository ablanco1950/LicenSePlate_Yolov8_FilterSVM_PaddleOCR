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
#print(Y_train)
X_train, Licenses=loadimages(dirname)
X_test, LicensesTest=loadimages(dirnameTest)


print("Number of imagenes to test : " + str(len(X_train)))
print("Number of  CodFilters  : " + str(len(Y_train)))

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


 #https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
model =  OneVsRestClassifier(SVC(kernel='linear', probability=True,  max_iter=1000)) #Creates model instance here 
#model =  OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, max_iter=1000)) #Creates model instance here
#model =  OneVsRestClassifier(SVC(kernel='poly', degree=8)) #Creates model instance here
#model =  OneVsRestClassifier(SVC(kernel='rbf')) 
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
   
   
    
    NumberImageOrder=NumberImageOrder+1
    CodFilter=predictions[i] 
    print(LicensesTest[i] + "  CodFilter = "+ str(CodFilter))
 
    
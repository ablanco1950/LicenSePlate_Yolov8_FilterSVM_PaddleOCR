Project that uses Yolov8 as license plate detector, followed by a filter whose code is obtained by applying a filter code to each of the filters considered in the reference project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR, with which you can recognize that board by paddleOCR and setting an SVM prediction

All the modules necessary for its execution can be installed, if the programs give a module not found error, by means of a simple pip.

The most important:

paddleocr must be installed (https://pypi.org/project/paddleocr/)

pip install paddleocr

yolo must be installed, if not, follow the instructions indicated in: https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

pip install ultralytics

As a previous step, the X_train and the Y_train that the SVM needs are created, the X_Train is the matrix of each image and the Y_train is made based on the code assigned (from 0 to 10) to the first filter with which paddleocr manages to recognize that license plate. of car in the reference project.

The Crea_Xtrain_Ytrain.py program is attached (its execution is not necessary), whose result after applying it to different image files (the input file is indicated in instruction 15) of renamed cars with their registration plate is saved in the Training folder , consisting of the image itself and a .txt file with the name of the car's license plate and containing the filter code assigned to that image. This Training file that is attached in .zip format is necessary to download it, like the Test.zip file and unzip them to run:

TrainCodFilterSVM.py


The result is the file with the model.pickle weights (because of its size I cannot upload it to github, but it takes a short time to obtain it) necessary to establish the predictions when executing the program:

GetNumberInternationalLicensePlate_Yolov8_SVMFilters_PaddleOCR_V1.py

Any folder may be tested changing instruction 14, the resultas in LicenseResults.txt file

Comparing with the reference project: https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR you get a lower precision but a considerable reduction in execution time.

The references are identical to those of the project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR:

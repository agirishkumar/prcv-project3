# prcv-project3
# Real-time 2-D Object Recognition

### Girish Kumar and Alexander Seljuk
Time travel days used: 2

This project explores binary image classification using threshholding, cleanup, segmentation, feature extraction, feature matching. Every object detected is classified given a database of features and labels. All the steps were implemented from the scratch.

Running the Program

## To compile and execute the program, follow these steps:

make

./prj3


## Functionality

The program is capturing a video stream from the input and will detect objects and show their precentage of oriented bounding box filled. 

Detection mode automatically classifies any number of found objects bigger than a certain pixel area. The classification is done by finding the scaled euclidian (closest neighbor)  or KNN with 2 neighbors depending on the classification mode. 

The database has the following object data: wallet, phone, mobile, gripper, controller, pan, headphones, vape. 

The labeling mode allows to label objects and remember them in the database. 

The confusion matrix will be built when the c key is pressed and user prompts the true label, it compares with with the detected label. when key is P is pressed it prints out the till then built confusion matrix and corresponding accuracy.

The program reacts to the following key presses:

`q` or `Esc` quit the program.

`n` or `N` train mode, asks the user for a label and saves the current object's features into the database.

`c` or `C` will request the correct image label to analyze if the classification was correct. (part of building confusion matrix)

`d` or `D` detect mode - will detect and label the regions.

`k` or `K` toggles between scaled euclidian and knn mode classification.

`p` or `P` prints the confusion matrix and the accuracy of the classification.



## Results:

The program successfully implements the pipeline and classifies objects on a video stream. It can apply different matching methods(knn and nearest neighbor). And automatically detect if an object is not known.

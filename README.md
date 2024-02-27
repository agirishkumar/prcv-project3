# prcv-project3
# Real-time 2-D Object Recognition
Girish Kumar and Alexander Seljuk

This project explores binary image classification using threshholding, cleanup, segmentation, feature extraction. The object's features are then collected into a database and new unknown objects are then classified using algorithms like knn.

Running the Program

## To compile and execute the program, follow these steps:

`make`

./Assignment3 


## Functionality

The program is capturing a video stream from the input and will detect objects and show their precentage of oriented bounding box filled. Detection mode automatically classifies any found objects bigger than a certain threshold. The classification is done by finding the closest neighbor or knn with 2 neighbors dependingon the classification mode. The database has the following object data: wallet, phone, mobile, gripper, controller, pan, headphones. The labeling mode allows to label objects and remember them in a database. The program reacts to the following key presses:

`q` quit the program.

`n` train mode, asks the user for a label and saves the current object's features into the database.

`c` will request the correct image label to analyze if the classification was correct.

`d` detect mode - will detect and label the regions.

`k` switch to knn mode classification, uses knn to classify objects.

`p` prints the confusion matrix.



## Results:

The program successfully classifies the known objects and detects unknown objects. 


## Time travel days: 2
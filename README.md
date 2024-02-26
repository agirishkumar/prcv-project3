# prcv-project3
# Real-time 2-D Object Recognition
Girish Kumar and Alexander Seljuk

This project explores binary image classification using threshholding, cleanup, segmentation, feature extraction. The object's features are then collected into a database and new unknown objects are then classified using algorithms like knn.

Running the Program

## To compile and execute the program, follow these steps:

`make`

./Assignment3 


## Functionality

The program is capturing a video stream from the input and will automatically classify any found objects bigger than a certain threshold, this is the detection mode. The database has the following object data: wallet, phone, mobile, gripper, controller, pan, headphones. The labeling mode allows to label objects and remember them in a database. The program also reacts to the following key presses:

`q` quit the program.

`n` train mode, asks the user for a label and saves the current object's features into the database.



## Results:

The program successfully classifies the known objects and marks unknow as unknown.

/*
Authors: Girish Kumar Adari, Alexander Seljuk
PRCV Project 3: Real-time 2-D Object Recognition

This is the main file for the image processing application.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h" // Ensure this file is updated with the new functions

using namespace std;
using namespace cv;

std::vector<std::string> getLabels() {
    // You should populate this list with the labels you are classifying
    return {"controller", "gripper", "watch", "wallet", "unknown"};
}

/**
 * The main function for the image processing application.
 * Lets the user choose between detecting and labeling objects.
 * 
 *
 * @return int the exit status of the program
 *
 * @throws None
 */
int main()
{
    cout << "Starting the image processing application..." << endl;

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Error: Cannot open the webcam stream." << endl;
        return -1;
    }

    // Define states for the application
    enum AppState
    {
        DETECTION,
        LABELING
    } state ;

    // Create windows
    namedWindow("Live Feed", WINDOW_AUTOSIZE);
    namedWindow("Preprocessed feed", WINDOW_AUTOSIZE);
    namedWindow("KNN thresholded feed", WINDOW_AUTOSIZE);
    namedWindow("Cleaned thresholded feed", WINDOW_AUTOSIZE);
    namedWindow("Region Map", WINDOW_AUTOSIZE);
    namedWindow("Filtered Region Map", WINDOW_AUTOSIZE);
    namedWindow("Feature Visualization", WINDOW_AUTOSIZE);

    Mat frame, preprocessedImg, kmeansImage, cleanedImage;
    string databaseFilename = "database.csv";
    string detectedLabel;

    std::map<std::string, std::map<std::string, int>> confusionMatrix;
    std::vector<std::string> labels = getLabels();
    initializeConfusionMatrix(confusionMatrix, labels);

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        imshow("Live Feed", frame);

        // Preprocess the frame
        preprocessImg(frame, preprocessedImg);
        imshow("Preprocessed feed", preprocessedImg);

        // Apply k-means thresholding
        kmeansThresholding(preprocessedImg, kmeansImage);
        imshow("KNN thresholded feed", kmeansImage);
        

        // Clean the thresholded image
        cleanThresholdedImage(kmeansImage, cleanedImage);
        imshow("Cleaned thresholded feed", cleanedImage);

        // Apply region growing on the cleaned, thresholded image
        Mat regionMap;
        regionGrowing(cleanedImage, regionMap);
        Mat coloredRegions = regionColor(regionMap);
        imshow("Region Map", coloredRegions);
        

        // Filter out small regions
        Mat filteredRegionMap = removeSmallRegions(regionMap, 8000);
        Mat coloredFilteredRegions = regionColor(filteredRegionMap);
        imshow("Filtered Region Map", coloredFilteredRegions);

        double minVal, maxVal;
            cv::minMaxLoc(regionMap, &minVal, &maxVal);
            int maxRegionID = static_cast<int>(maxVal);

            // Compute and display features for each major region
            for (int regionID = 1; regionID <= maxRegionID; ++regionID)
            {
                RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
                if (features.area > 10000)
                {
                    vector<Coordinate> obb = calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y);
                    drawObb(frame, obb);
                }
            }
        

        if (state == DETECTION)
        {
            // Your detection code goes here
            detectedLabel = detectAndLabelRegions(frame, filteredRegionMap, databaseFilename);

            
        }
            imshow("Feature Visualization", frame);


        // Handle user input
        char key = (char)waitKey(1);
        if (key == 'n' || key == 'N')
        {
            // toggle between states
            state = (state == DETECTION) ? LABELING : DETECTION;
            if (state == LABELING)
            {
                cout << "Enter label for the current object: ";
                string label;
                cin >> label; // Get label from the user

                // Compute features for the region
                RegionFeatures features = computeRegionFeatures(filteredRegionMap, 1); // Adjust the region ID if necessary

                if (features.area > 1000)
                {
                    saveFeatureVectorToFile(features, label, databaseFilename);
                    cout << "Feature vector saved successfully. Switching back to detection mode." << endl;
                    state = DETECTION;
                }
                else
                {
                    cout << "Selected region is too small and was not saved." << endl;
                }
            }
        }
        else if(key == 'c' || key == 'C'){
            if (state == DETECTION){
                cout<< "Enter the true label for the current object: ";
                string trueLabel;
                cin >> trueLabel;
                updateConfusionMatrix(confusionMatrix, trueLabel, detectedLabel);
                if(trueLabel == detectedLabel){
                    cout << "Correctly identified the object." << endl;
                
                }else{
                    cout << "Incorrectly identified the object. Switching back to labeling mode." << endl;
                    
                }
                
            }
            state = DETECTION;

        }
        else if (key == 'p' || key == 'P')
        {
            cout << "Confusion Matrix:" << endl;
            printConfusionMatrix(confusionMatrix);

            cout << "Accuracy: " << calculateAccuracy(confusionMatrix) << endl;
        }else if (key == 'd' || key == 'D')
        {
            state = DETECTION;
        }
        else if (key == 27 || key == 'q')
        {
            break; // Exit if ESC or 'q' is pressed
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}


// int main() {
//     std::cout << "Starting the image processing application..." << endl;

//     // VideoCapture cap(0);
//     // if (!cap.isOpened()) {
//     //     cerr << "Error: Cannot open the webcam stream." << endl;
//     //     return -1;
//     // }
//     Mat frame = imread("sp1.jpeg");
//     // Define states for the application
//     enum AppState { DETECTION, LABELING } state = DETECTION;

//     // Create windows
//     namedWindow("Live Feed", WINDOW_AUTOSIZE);
//     namedWindow("Feature Visualization", WINDOW_AUTOSIZE);

//     Mat preprocessedImg, kmeansImage, cleanedImage;
//     string databaseFilename = "database.csv";

//         imshow("Live Feed", frame);

//         // Preprocess the frame
//         preprocessImg(frame, preprocessedImg);

//         // Apply k-means thresholding
//         kmeansThresholding(preprocessedImg, kmeansImage);

//         // Clean the thresholded image
//         cleanThresholdedImage(kmeansImage, cleanedImage);

//         // Apply region growing on the cleaned, thresholded image
//         Mat regionMap;
//         regionGrowing(cleanedImage, regionMap);

//         // Filter out small regions
//         Mat filteredRegionMap = removeSmallRegions(regionMap, 5000);

//             // Your detection code goes here
//             detectAndLabelRegions(frame, filteredRegionMap, databaseFilename);

//             double minVal, maxVal;
//             cv::minMaxLoc(regionMap, &minVal, &maxVal);
//             int maxRegionID = static_cast<int>(maxVal);

//             // Compute and display features for each major region
//             for (int regionID = 1; regionID <= maxRegionID; ++regionID) {
//                 RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
//                 if (features.area > 10000) {
//                 vector<Coordinate> obb = calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y);
//                 drawObb(frame, obb); 
//             }
//             }
//             imshow("Feature Visualization", frame);
//             waitKey(0);

//         // Handle user input
//             // toggle between states
//                 std::cout << "Enter label for the current object: ";
//                 string label;
//                 std::cin >> label;  // Get label from the user

//                 // Compute features for the region
//                 RegionFeatures features = computeRegionFeatures(filteredRegionMap, 1);  // Adjust the region ID if necessary

//                 if (features.area > 1000) {
//                     saveFeatureVectorToFile(features, label, databaseFilename);
//                     std::cout << "Feature vector saved successfully. Switching back to detection mode." << endl;
//                     state = DETECTION;
//                 } else {
//                     std::cout << "Selected region is too small and was not saved." << endl;
//                 }

//     return 0;
// }
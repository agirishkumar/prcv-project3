// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>
// #include "helpers.h" // Ensure this file is updated with the new functions

// using namespace std;
// using namespace cv;

// int main() {
//     cout << "Starting the image processing application..." << endl;

//     VideoCapture cap(2); 
//     if (!cap.isOpened()) {
//         cerr << "Error: Cannot open the webcam stream." << endl;
//         return -1;
//     }

//     // Create a window to display the video
//     namedWindow("Live Feed", WINDOW_AUTOSIZE);

    
//     while (true) {
//         Mat frame;
//         cap >> frame;
//         if (frame.empty()) break;

//         imshow("Live Feed", frame);

//         Mat preprocessedImg, kmeansImage, cleanedImage, featureVisualization;

//         // Preprocess the image
//         if (preprocessImg(frame, preprocessedImg) != 0) {
//             cerr << "Error during image preprocessing." << endl;
//             return -1;
//         }

//         imshow("Preprocessed Video", preprocessedImg);

//         // Apply k-means thresholding
//         if (kmeansThresholding(preprocessedImg, kmeansImage) != 0) {
//             cerr << "Error during K-means thresholding." << endl;
//             return -1;
//         }

//         imshow("Kmeans Thresholded Video", kmeansImage);

//         // Clean the thresholded image
//         cleanThresholdedImage(kmeansImage, cleanedImage);

//         imshow("Cleaned Thresholded Video", cleanedImage);


//         // Apply region growing on the cleaned, thresholded image
//         Mat regionMap;
//         int regionCount = regionGrowing(cleanedImage, regionMap);
//         // cout << "Region count before filtering: " << regionCount << endl;

//         // Visualize initial segmentation
//         Mat coloredMap = regionColor(regionMap);

//         namedWindow("Initial Segmentation", WINDOW_AUTOSIZE);
//         imshow("Initial Segmentation", coloredMap);

//         // Filter out small regions and visualize the result
//         Mat filteredRegionMap = removeSmallRegions(regionMap, 200); // Adjust the size threshold as needed

//         // Count and print the region count after filtering
//         double minVal, maxVal;
//         minMaxLoc(filteredRegionMap, &minVal, &maxVal); // Finds the min and max pixel values and their positions
//         int filteredRegionCount = static_cast<int>(maxVal);
//         // cout << "Region count after filtering: " << filteredRegionCount << endl;

//         // Visualize the filtered segmentation
//         Mat filteredColoredMap = regionColor(filteredRegionMap);

//         namedWindow("Filtered Segmentation", WINDOW_AUTOSIZE);
//         imshow("Filtered Segmentation", filteredColoredMap);

//         // Compute and display features for each major region
//         for (int regionID = 1; regionID <= filteredRegionCount; ++regionID) {
//             RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
//             cout << "area: "<< features.area << endl;
//             if (features.area > 5000) {
//         drawObb(frame, calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y));
//             }
            
//         }

        
        
//         namedWindow("Feature Visualization", WINDOW_AUTOSIZE); // Additional window for feature visualization    
//         imshow("Feature Visualization", frame); // Show src again with features overlaid


//         // Wait for 'N' key press to trigger feature extraction and saving
//         char key = (char)waitKey(1); // Wait for a key press
//         if (key == 'n' || key == 'N') {
//             string label;
//             cout << "Enter label for the current object: ";
//             cin >> label; // Get label from the user

//             // Assume regionID 1 is of interest for simplicity, adjust according to your application
//             RegionFeatures features = computeRegionFeatures(filteredRegionMap, 1);
//             saveFeatureVectorToFile(features, label, "database.csv");

//             cout << "Feature vector saved successfully." << endl;
//         } else if (key == 27 || key == 'q') { // ESC or 'q' key to exit
//             break;
//         }      

//     }

//     cap.release();
//     destroyAllWindows();
    

//     return 0;
// }


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h"  // Ensure this file is updated with the new functions

using namespace std;
using namespace cv;

int main() {
    cout << "Starting the image processing application..." << endl;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open the webcam stream." << endl;
        return -1;
    }

    // Define states for the application
    enum AppState { DETECTION, LABELING } state = DETECTION;

    // Create windows
    namedWindow("Live Feed", WINDOW_AUTOSIZE);
    namedWindow("Feature Visualization", WINDOW_AUTOSIZE);

    Mat frame, preprocessedImg, kmeansImage, cleanedImage;
    string databaseFilename = "database.csv";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        imshow("Live Feed", frame);

        // Preprocess the frame
        preprocessImg(frame, preprocessedImg);

        // Apply k-means thresholding
        kmeansThresholding(preprocessedImg, kmeansImage);

        // Clean the thresholded image
        cleanThresholdedImage(kmeansImage, cleanedImage);

        // Apply region growing on the cleaned, thresholded image
        Mat regionMap;
        regionGrowing(cleanedImage, regionMap);

        // Filter out small regions
        Mat filteredRegionMap = removeSmallRegions(regionMap, 5000);

        if (state == DETECTION) {
            // Your detection code goes here
            detectAndLabelRegions(frame, filteredRegionMap, databaseFilename);

            double minVal, maxVal;
            cv::minMaxLoc(regionMap, &minVal, &maxVal);
            int maxRegionID = static_cast<int>(maxVal);

            // Compute and display features for each major region
            for (int regionID = 1; regionID <= maxRegionID; ++regionID) {
                RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
                if (features.area > 10000) {
                vector<Coordinate> obb = calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y);
                drawObb(frame, obb); 
            }
            }
            imshow("Feature Visualization", frame);
        }

        // Handle user input
        char key = (char)waitKey(1);
        if (key == 'n' || key == 'N') {
            // toggle between states
            state = (state == DETECTION) ? LABELING : DETECTION;
            if (state == LABELING) {
                cout << "Enter label for the current object: ";
                string label;
                cin >> label;  // Get label from the user

                // Compute features for the region
                RegionFeatures features = computeRegionFeatures(filteredRegionMap, 1);  // Adjust the region ID if necessary

                if (features.area > 1000) {
                    saveFeatureVectorToFile(features, label, databaseFilename);
                    cout << "Feature vector saved successfully. Switching back to detection mode." << endl;
                    state = DETECTION;
                } else {
                    cout << "Selected region is too small and was not saved." << endl;
                }
            }
        } else if (key == 27 || key == 'q') {
            break;  // Exit if ESC or 'q' is pressed
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
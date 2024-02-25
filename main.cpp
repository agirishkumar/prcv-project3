#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h" // Ensure this file is updated with the new functions

using namespace std;
using namespace cv;

int main() {
    cout << "Starting the image processing application..." << endl;

    VideoCapture cap(2); 
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open the webcam stream." << endl;
        return -1;
    }

    // Create a window to display the video
    namedWindow("Live Feed", WINDOW_AUTOSIZE);

    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        imshow("Live Feed", frame);

        Mat preprocessedImg, kmeansImage, cleanedImage, featureVisualization;

        // Preprocess the image
        if (preprocessImg(frame, preprocessedImg) != 0) {
            cerr << "Error during image preprocessing." << endl;
            return -1;
        }

        imshow("Preprocessed Video", preprocessedImg);

        // Apply k-means thresholding
        if (kmeansThresholding(preprocessedImg, kmeansImage) != 0) {
            cerr << "Error during K-means thresholding." << endl;
            return -1;
        }

        imshow("Kmeans Thresholded Video", kmeansImage);

        // Clean the thresholded image
        cleanThresholdedImage(kmeansImage, cleanedImage);

        imshow("Cleaned Thresholded Video", cleanedImage);


        // Apply region growing on the cleaned, thresholded image
        Mat regionMap;
        int regionCount = regionGrowing(cleanedImage, regionMap);
        // cout << "Region count before filtering: " << regionCount << endl;

        // Visualize initial segmentation
        Mat coloredMap = regionColor(regionMap);

        namedWindow("Initial Segmentation", WINDOW_AUTOSIZE);
        imshow("Initial Segmentation", coloredMap);

        // Filter out small regions and visualize the result
        Mat filteredRegionMap = removeSmallRegions(regionMap, 200); // Adjust the size threshold as needed

        // Count and print the region count after filtering
        double minVal, maxVal;
        minMaxLoc(filteredRegionMap, &minVal, &maxVal); // Finds the min and max pixel values and their positions
        int filteredRegionCount = static_cast<int>(maxVal);
        // cout << "Region count after filtering: " << filteredRegionCount << endl;

        // Visualize the filtered segmentation
        Mat filteredColoredMap = regionColor(filteredRegionMap);

        namedWindow("Filtered Segmentation", WINDOW_AUTOSIZE);
        imshow("Filtered Segmentation", filteredColoredMap);

        // Compute and display features for each major region
        for (int regionID = 1; regionID <= filteredRegionCount; ++regionID) {
            RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
            cout << "area: "<< features.area << endl;
            if (features.area > 5000) {
        drawObb(frame, calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y));
            }
            
        }

        
        
        namedWindow("Feature Visualization", WINDOW_AUTOSIZE); // Additional window for feature visualization    
        imshow("Feature Visualization", frame); // Show src again with features overlaid


        // Wait for 'N' key press to trigger feature extraction and saving
        char key = (char)waitKey(1); // Wait for a key press
        if (key == 'n' || key == 'N') {
            string label;
            cout << "Enter label for the current object: ";
            cin >> label; // Get label from the user

            // Assume regionID 1 is of interest for simplicity, adjust according to your application
            RegionFeatures features = computeRegionFeatures(filteredRegionMap, 1);
            saveFeatureVectorToFile(features, label, "database.csv");

            cout << "Feature vector saved successfully." << endl;
        } else if (key == 27 || key == 'q') { // ESC or 'q' key to exit
            break;
        }      

    }

    cap.release();
    destroyAllWindows();
    

    return 0;
}


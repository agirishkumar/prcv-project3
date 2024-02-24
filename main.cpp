#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h" // Ensure this file is updated with the new functions

using namespace std;
using namespace cv;

int main() {
    cout << "Starting the image processing application..." << endl;

    Mat src = imread("sample3.jpg");
    if (src.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", src);

    Mat preprocessedImg, binaryImage, kmeansImage, cleanedImage;

    // Preprocess the image
    if (preprocessImg(src, preprocessedImg) != 0) {
        cerr << "Error during image preprocessing." << endl;
        return -1;
    }

    namedWindow("Preprocessed Image", WINDOW_AUTOSIZE);
    imshow("Preprocessed Image", preprocessedImg);
  
    // Apply binary thresholding
    if (binaryThresholding(preprocessedImg, binaryImage) != 0) {
        cerr << "Error during binary thresholding." << endl;
        return -1;
    }

    namedWindow("Binary thresholded Image", WINDOW_AUTOSIZE);
    imshow("Binary thresholded Image", binaryImage);
  
    // Apply k-means thresholding
    if (kmeansThresholding(preprocessedImg, kmeansImage) != 0) {
        cerr << "Error during K-means thresholding." << endl;
        return -1;
    }

    namedWindow("Kmeans thresholded Image", WINDOW_AUTOSIZE);
    imshow("Kmeans thresholded Image", kmeansImage);
  
    cleanThresholdedImage(kmeansImage, cleanedImage);

    namedWindow("Cleaned Image", WINDOW_AUTOSIZE);
    imshow("Cleaned Image", cleanedImage);

    // Apply region growing on the cleaned, thresholded image
    Mat regionMap;
    int regionCount = regionGrowing(cleanedImage, regionMap);
    cout << "Region count before filtering: " << regionCount << endl;

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
    cout << "Region count after filtering: " << filteredRegionCount << endl;

    // Visualize the filtered segmentation
    Mat filteredColoredMap = regionColor(filteredRegionMap);

    namedWindow("Filtered Segmentation", WINDOW_AUTOSIZE);
    imshow("Filtered Segmentation", filteredColoredMap);

    // Compute and display features for each major region
    // Assuming you have defined computeRegionFeatures and displayRegionFeatures functions
    for (int regionID = 1; regionID <= filteredRegionCount; ++regionID) {
        RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);
        drawObb(src, calculateOrientedBoundingBox(filteredRegionMap, regionID, features.theta, features.centroid.x, features.centroid.y));
    }

    
    
    namedWindow("Feature Visualization", WINDOW_AUTOSIZE); // Additional window for feature visualization    
    imshow("Feature Visualization", src); // Show src again with features overlaid

    cout << "Press 't' or 'T' to enter training mode and label the current object." << endl;

    char key = waitKey(0); // Wait for a key press
    if (key == 't' || key == 'T') {
        cout << "Training mode activated. Enter label for the current object: ";
        string label;
        cin >> label; // Get label from the user

        // Assuming you've identified a regionID to compute features for
        int regionID = 1; // Placeholder: adapt this to your method of selecting a region
        RegionFeatures features = computeRegionFeatures(filteredRegionMap, regionID);

        // Save the feature vector and label to a file
        if (saveFeatureVectorToFile(features, label, "training_data.csv")) {
            cout << "Training data saved successfully." << endl;
        } else {
            cerr << "Error saving training data." << endl;
        }
    } else {
        cout << "Continuing without entering training mode." << endl;
    }

    return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h"

using namespace std;
using namespace cv;

int main() {
  cout << "Starting the image processing application..." << endl;

  Mat src = imread("sample2.jpg");

  if (src.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

  Mat preprocessedImg, binaryImage, kmeansImage, cleanedImage;

  // Preprocess the image
  if (preprocessImg(src, preprocessedImg) != 0) {
      cerr << "Error during image preprocessing." << endl;
      return -1;
  }
  
  // Apply binary thresholding
  if (binaryThresholding(preprocessedImg, binaryImage) != 0) {
      cerr << "Error during binary thresholding." << endl;
      return -1;
  }
  
  // Apply k-means thresholding
  if (kmeansThresholding(preprocessedImg, kmeansImage) != 0) {
      cerr << "Error during K-means thresholding." << endl;
      return -1;
  }
  

  cleanThresholdedImage(kmeansImage, cleanedImage);

  // Apply region growing on the cleaned, thresholded image
  Mat regionMap;
  int regionCount = regionGrowing(cleanedImage, regionMap);
  cout << "Region count before filtering: " << regionCount << endl;

  // Visualize initial segmentation
  Mat coloredMap = regionColor(regionMap);

  // Filter out small regions and visualize the result
  Mat filteredRegionMap = removeSmallRegions(regionMap, 200); 

  // Count and print the region count after filtering
  double minVal, maxVal;
  minMaxLoc(filteredRegionMap, &minVal, &maxVal); // Finds the min and max pixel values and their positions
  int filteredRegionCount = static_cast<int>(maxVal);
  cout << "Region count after filtering: " << filteredRegionCount << endl;

  // Visualize the filtered segmentation
  Mat filteredColoredMap = regionColor(filteredRegionMap);  

  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("Preprocessed Image", WINDOW_AUTOSIZE);
  namedWindow("Binary thresholded Image", WINDOW_AUTOSIZE);
  namedWindow("Kmeans thresholded Image", WINDOW_AUTOSIZE);
  namedWindow("Cleaned Image", WINDOW_AUTOSIZE);
  namedWindow("Initial Segmentation", WINDOW_AUTOSIZE);
  namedWindow("Filtered Segmentation", WINDOW_AUTOSIZE);

  imshow("Original Image", src);
  imshow("Preprocessed Image", preprocessedImg);
  imshow("Binary thresholded Image", binaryImage);
  imshow("Kmeans thresholded Image", kmeansImage);
  imshow("Cleaned Image", cleanedImage);
  imshow("Initial Segmentation", coloredMap);
  imshow("Filtered Segmentation", filteredColoredMap);

  waitKey(0); 

  return 0;
}

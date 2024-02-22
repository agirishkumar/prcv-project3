#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helpers.h"

using namespace std;
using namespace cv;

int main() {
  cout << "Hello, World!" << endl;

  Mat src = imread("Proj03Examples/img1p3.png");
  Mat dst, binaryImage, kmeansImage, cleanImage;
  preprocessImg(src, dst);
  binaryThresholding(dst,binaryImage);
  kmeansThresholding(dst, kmeansImage);

  cleanThresholdedImage(kmeansImage, cleanImage);

  namedWindow("Image", WINDOW_AUTOSIZE);
  namedWindow("preprocessed Image", WINDOW_AUTOSIZE);
  namedWindow("binary thresholded Image", WINDOW_AUTOSIZE);
  namedWindow("kmeans thresholded Image", WINDOW_AUTOSIZE);
  namedWindow("cleaned Image", WINDOW_AUTOSIZE);

  imshow("Image", src);
  imshow("preprocessed Image", dst);
  imshow("binary thresholded Image", binaryImage);
  imshow("kmeans thresholded Image", kmeansImage);
  imshow("cleaned Image", cleanImage);

  waitKey(0);

  Mat image = imread("images/img5P3.png");

  Mat grey;
  cvtColor(image, grey, COLOR_BGR2GRAY);
  Mat thresholded;
  imshow("image", grey);
  waitKey(0);
  threshold(grey, thresholded, 90, 255, THRESH_BINARY_INV);
  imshow("image", thresholded);
  waitKey(0);
  Mat regionMap;
  int regionCount = regionGrowing(src, thresholded, regionMap);
  // cout << regionCount;
  Mat coloredMap = regionColor(regionMap);
  imshow("image", coloredMap);
  waitKey(0);
  Mat newRegion = removeSmallRegions(regionMap, 2000);
  coloredMap = regionColor(newRegion);
  imshow("image", newRegion);
  waitKey(0);
  return 0;

  return 0;
}
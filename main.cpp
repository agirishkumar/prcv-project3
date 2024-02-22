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

  return 0;
}
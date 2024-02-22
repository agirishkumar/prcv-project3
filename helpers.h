#ifndef MHELPERS_H
#define MHELPERS_H

#include <opencv2/opencv.hpp>
using namespace cv;

// Preprocesses an input image by converting it to grayscale and applying a contrast-limited adaptive histogram equalization (CLAHE).
// Parameters:
//   const cv::Mat &src: Source image to preprocess.
//   cv::Mat &preprocessedImg: Destination image that will hold the preprocessed image.
// Returns:
//   int: 0 on success, -1 if the source image is empty.
int preprocessImg(const cv::Mat &src, cv::Mat &preprocessedImg);

// Applies a 5x5 Gaussian blur to an image using a naive approach with a fixed kernel.
// Parameters:
//   const cv::Mat &src: Source image to apply the Gaussian blur.
//   cv::Mat &dst: Destination image that will hold the blurred image.
// Returns:
//   int: 0 on success, -1 if the source image is empty.
int guassianBlur5x5(const cv::Mat &src, cv::Mat &dst);

// Performs binary thresholding on an image.
// Parameters:
//   const cv::Mat &src: Source image to threshold.
//   cv::Mat &dst: Destination image that will hold the thresholded image.
// Returns:
//   int: 0 on success.
int binaryThresholding(const cv::Mat &src, cv::Mat &dst);

// Performs k-means thresholding on an image to determine an optimal binary threshold.
// Parameters:
//   const cv::Mat &src: Source image to threshold.
//   cv::Mat &dst: Destination image that will hold the thresholded image.
// Returns:
//   int: 0 on success.
int kmeansThresholding(const cv::Mat &src, cv::Mat &dst);

// Creates a rectangular structuring element (kernel) of the specified size.
// Parameters:
//   int width: Width of the kernel.
//   int height: Height of the kernel.
// Returns:
//   cv::Mat: The created rectangular structuring element.
Mat rectangularKernel(int width, int height);

// Cleans a thresholded image using morphological operations to reduce noise and fill holes.
// Parameters:
//   cv::Mat &src: Source image that has been thresholded.
//   cv::Mat &dst: Destination image that will hold the cleaned image.
void cleanThresholdedImage(cv::Mat &src, cv::Mat &dst);

// Performs morphological closing on an image (dilation followed by erosion).
// Parameters:
//   const cv::Mat &src: Source image to apply closing.
//   cv::Mat &dst: Destination image that will hold the result of closing.
//   const cv::Mat &kernel: The structuring element used for closing.
void closing(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

// Performs morphological opening on an image (erosion followed by dilation).
// Parameters:
//   const cv::Mat &src: Source image to apply opening.
//   cv::Mat &dst: Destination image that will hold the result of opening.
//   const cv::Mat &kernel: The structuring element used for opening.
void opening(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

// Dilates an image using a given structuring element.
// Parameters:
//   const cv::Mat &src: Source image to apply dilation.
//   cv::Mat &dst: Destination image that will hold the dilated image.
//   const cv::Mat &kernel: The structuring element used for dilation.
void dilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

// Erodes an image using a given structuring element.
// Parameters:
//   const cv::Mat &src: Source image to apply erosion.
//   cv::Mat &dst: Destination image that will hold the eroded image.
//   const cv::Mat &kernel: The structuring element used for erosion.
void erode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

/**
   Region growing algorithm to find regions and give them ids.
   @param Mat map - a map where each pixel is labeled 255 - foreground, 0 - background.
   @param Mat regionMap - the destination map with pixels labeld with their region ids and 0 - background.
   @return int number of regions.
*/
int regionGrowing(Mat &source, Mat &map, Mat &regionMap);

/**
   Creates a colored visualization of the region map.
   @param Mat regionMap - a mat of type 8UC1 with each pixel's region id
   @return Mat - 8UC3 Mat with colors assigned to each region.
*/
Mat regionColor(Mat &regionMap);

/**
    Removes regions smaller than minSize number of pixels.
    @param Mat regionMap - a mat of type 8UC1 with each pixel's region id.
    @param int minSize - the minimum pixel size to be considered a region.
    @return Mat - 8UC3 Mat with colors assigned to each region.
*/
Mat removeSmallRegions(Mat &regionMap, int minSize);

#endif // MHELPERS_H

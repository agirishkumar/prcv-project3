#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "helpers.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>

using namespace cv;
using namespace std;

int guassianBlur5x5(const Mat &src, Mat &dst)
{
  if (src.empty())
  {
    cout << "Source image is empty!" << endl;
    return -1; // Return an error code if empty
  }

  dst = src.clone();

  int gaussianKernel[5][5] = {
      {1, 4, 7, 4, 1},
      {4, 16, 26, 16, 4},
      {7, 26, 41, 26, 7},
      {4, 16, 26, 16, 4},
      {1, 4, 7, 4, 1}};

  int kernelSum = 273;

  for (int y = 2; y < src.rows - 2; ++y)
  {
    for (int x = 2; x < src.cols - 2; ++x)
    {
      float sumB = 0, sumG = 0, sumR = 0;

      for (int dy = -2; dy <= 2; ++dy)
      {
        for (int dx = -2; dx <= 2; ++dx)
        {
          Vec3b pixel = src.at<Vec3b>(y + dy, x + dx);
          int kernelValue = gaussianKernel[dy + 2][dx + 2];
          sumB += pixel[0] * kernelValue;
          sumG += pixel[1] * kernelValue;
          sumR += pixel[2] * kernelValue;
        }
      }

      // Normalize and assign blurred pixel values
      dst.at<Vec3b>(y, x) = Vec3b(static_cast<uchar>(sumB / kernelSum), static_cast<uchar>(sumG / kernelSum), static_cast<uchar>(sumR / kernelSum));
    }
  }

  return 0;
}

/**
 * Preprocesses the input image by applying custom Gaussian blur, converting to grayscale, and enhancing contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).
 *
 * @param src the input image to be preprocessed
 * @param preprocessedImg the preprocessed output image
 *
 * @return 0 on success, -1 on failure
 *
 * @throws None
 */
int preprocessImg(const cv::Mat &src, cv::Mat &preprocessedImg)
{
  if (src.empty())
  {
    cout << "Source image is empty!" << endl;
    return -1;
  }

  // Apply custom Gaussian blur
  Mat blurredImg;
  if (guassianBlur5x5(src, blurredImg) != 0)
  {
    cout << "Error applying Gaussian blur." << endl;
    return -1;
  }

  // Convert to grayscale
  Mat grayImg;
  cvtColor(blurredImg, grayImg, COLOR_BGR2GRAY);

  // Enhance contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit(2.0);
  clahe->setTilesGridSize(Size(8, 8));
  clahe->apply(grayImg, preprocessedImg);

  return 0; // Success
}

// binary thresholding
int binaryThresholding(const Mat &src, Mat &dst)
{
  // Convert the source image to grayscale if it is not already
  Mat grayImage;
  if (src.channels() == 3)
  {
    cvtColor(src, grayImage, COLOR_BGR2GRAY);
  }
  else
  {
    grayImage = src.clone();
  }

  // Create the destination image with the same size as source; initialize to zero
  dst = Mat::zeros(grayImage.size(), grayImage.type());

  double thresh = 150; // threshold value
  uchar maxval = 255;  // Maximum intensity value for binary output

  // Iterate through the image and applying the binary thresholding
  for (int y = 0; y < grayImage.rows; ++y)
  {
    for (int x = 0; x < grayImage.cols; ++x)
    {
      // Get the current pixel value
      uchar pixelValue = grayImage.at<uchar>(y, x);

      if (pixelValue >= thresh)
      {
        dst.at<uchar>(y, x) = maxval;
      }
      else
      {
        dst.at<uchar>(y, x) = 0;
      }
    }
  }

  return 0;
}

/**
 * Performs K-means thresholding on the input image.
 *
 * @param src input image
 * @param dst thresholded output image
 *
 * @return 0 if successful
 *
 * @throws None
 */
int kmeansThresholding(const Mat &src, Mat &dst)
{
  Mat gray;
  if (src.channels() == 3)
  {
    cvtColor(src, gray, COLOR_BGR2GRAY);
  }
  else
  {
    gray = src.clone();
  }

  const int maxIterations = 10;
  const double epsilon = 1.0;
  bool converged = false;
  int iterations = 0;

  // Initialize means randomly
  double mean1 = 0, mean2 = 255;
  while (!converged && iterations < maxIterations)
  {
    vector<double> cluster1, cluster2;

    // Assign pixels to the nearest cluster
    for (int y = 0; y < gray.rows; y++)
    {
      for (int x = 0; x < gray.cols; x++)
      {
        double pixelValue = static_cast<double>(gray.at<uchar>(y, x));
        if (abs(pixelValue - mean1) < abs(pixelValue - mean2))
        {
          cluster1.push_back(pixelValue);
        }
        else
        {
          cluster2.push_back(pixelValue);
        }
      }
    }

    // Recalculate means
    double newMean1 = accumulate(cluster1.begin(), cluster1.end(), 0.0) / cluster1.size();
    double newMean2 = accumulate(cluster2.begin(), cluster2.end(), 0.0) / cluster2.size();

    // Check for convergence
    if (abs(newMean1 - mean1) < epsilon && abs(newMean2 - mean2) < epsilon)
    {
      converged = true;
    }
    else
    {
      mean1 = newMean1;
      mean2 = newMean2;
    }

    iterations++;
  }

  // Compute threshold as the mean of the two cluster centers
  double threshold = (mean1 + mean2) / 2.0;

  // Apply threshold to create binary image
  dst = Mat::zeros(gray.size(), gray.type());
  for (int y = 0; y < gray.rows; y++)
  {
    for (int x = 0; x < gray.cols; x++)
    {
      if (gray.at<uchar>(y, x) > threshold)
      {
        dst.at<uchar>(y, x) = 255; // Above threshold
      }
      else
      {
        dst.at<uchar>(y, x) = 0; // Below threshold
      }
    }
  }

  return 0;
}

/**
 * Creates a rectangular kernel of the specified width and height.
 *
 * @param width the width of the rectangular kernel
 * @param height the height of the rectangular kernel
 *
 * @return the rectangular kernel as a Mat object
 *
 * @throws ErrorType any potential errors that may occur
 */
Mat rectangularKernel(int width, int height)
{
  return Mat::ones(Size(width, height), CV_8U);
}

/**
 * Erodes the input image using the specified kernel.
 *
 * @param src the input image
 * @param dst the output eroded image
 * @param kernel the structuring element used for erosion
 *
 * @return void
 *
 * @throws None
 */
void erode(const Mat &src, Mat &dst, const Mat &kernel)
{
  dst = src.clone();
  int kernelRadiusX = kernel.size().width / 2;
  int kernelRadiusY = kernel.size().height / 2;

  for (int y = kernelRadiusY; y < src.rows - kernelRadiusY; ++y)
  {
    for (int x = kernelRadiusX; x < src.cols - kernelRadiusX; ++x)
    {
      uchar minVal = 255;
      for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky)
      {
        for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx)
        {
          if (kernel.at<uchar>(ky + kernelRadiusY, kx + kernelRadiusX))
          {
            minVal = min(minVal, src.at<uchar>(y + ky, x + kx));
          }
        }
      }
      dst.at<uchar>(y, x) = minVal;
    }
  }
}

/**
 * Dilates the input image using the specified kernel.
 *
 * @param src input image
 * @param dst output image
 * @param kernel structuring element used for dilation
 *
 * @return None
 *
 * @throws None
 */
void dilate(const Mat &src, Mat &dst, const Mat &kernel)
{
  dst = src.clone();
  int kernelRadiusX = kernel.size().width / 2;
  int kernelRadiusY = kernel.size().height / 2;

  for (int y = kernelRadiusY; y < src.rows - kernelRadiusY; ++y)
  {
    for (int x = kernelRadiusX; x < src.cols - kernelRadiusX; ++x)
    {
      uchar maxVal = 0;
      for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky)
      {
        for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx)
        {
          if (kernel.at<uchar>(ky + kernelRadiusY, kx + kernelRadiusX))
          {
            maxVal = max(maxVal, src.at<uchar>(y + ky, x + kx));
          }
        }
      }
      dst.at<uchar>(y, x) = maxVal;
    }
  }
}

/**
 * Perform opening operation on the input image using the specified kernel.
 *
 * @param src the input image
 * @param dst the destination image where the result will be stored
 * @param kernel the structuring element used for the opening operation
 *
 * @return void
 *
 * @throws N/A
 */
void opening(const Mat &src, Mat &dst, const Mat &kernel)
{
  Mat temp;
  erode(src, temp, kernel);
  dilate(temp, dst, kernel);
}

/**
 * Perform closing operation on the input image using the provided kernel.
 *
 * @param src input image
 * @param dst output image
 * @param kernel structuring element for the closing operation
 *
 * @return void
 *
 * @throws N/A
 */
void closing(const Mat &src, Mat &dst, const Mat &kernel)
{
  Mat temp;
  dilate(src, temp, kernel);
  erode(temp, dst, kernel);
}

/**
 * Cleans the thresholded image by applying opening and closing operations.
 *
 * @param src the input thresholded image
 * @param dst the output cleaned image
 *
 * @return void
 *
 * @throws N/A
 */
void cleanThresholdedImage(Mat &src, Mat &dst)
{

  // Create a 3x3 rectangular structuring element
  Mat kernel = rectangularKernel(3, 3);

  // Apply opening to remove small noise
  Mat opened;
  opening(src, opened, kernel);

  // Apply closing to close small holes
  closing(opened, dst, kernel);
}

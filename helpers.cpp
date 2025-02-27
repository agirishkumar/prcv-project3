/*
Authors: Girish Kumar Adari, Alexander Seljuk
PRCV Project 3: Real-time 2-D Object Recognition

this file contains all the helper functions for the image processing and application.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "helpers.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <stack>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

/**
 * Performs a 5x5 Gaussian blur on the input image.
 *
 * @param src input image to be blurred
 * @param dst output image with Gaussian blur applied
 *
 * @return 0 if successful, -1 if the input image is empty
 *
 * @throws None
 */
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
      if (gray.at<uchar>(y, x) <= threshold)
      {
        dst.at<uchar>(y, x) = 255; // Below or equal to threshold
      }
      else
      {
        dst.at<uchar>(y, x) = 0; // Above threshold
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

/**
   Region growing algorithm to find regions and give them ids.
   @param Mat map - a map where each pixel is labeled 255 - foreground, 0 - background.
   @param Mat regionMap - the destination map with pixels labeld with their region ids and 0 - background.
   @return int number of regions.
*/
int regionGrowing(Mat &map, Mat &regionMap)
{

  stack<tuple<int, int>> pixels;
  Mat seen = Mat::zeros(map.size(), CV_8UC1);
  regionMap = Mat::zeros(map.size(), CV_8UC1);
  int regionIndex = 1;
  // Iterate throught the image.
  for (int i = 0; i < map.rows; i++)
  {
    for (int j = 0; j < map.cols; j++)
    {
      // If a pixel is foreground then push it to the stack.
      if (map.ptr<uchar>(i)[j] == 255 && regionMap.ptr<uchar>(i)[j] == 0)
      {
        tuple<int, int> pixel = make_tuple(i, j);
        pixels.push(pixel);
        // pop pixel and assign current region, for each pixels neighbor check if it is a foreground and push it to stack
        while (!pixels.empty())
        {
          auto [x, y] = pixels.top();
          pixels.pop();

          regionMap.ptr<uchar>(x)[y] = regionIndex;

          auto checkAndPush = [&](int nx, int ny)
          {
            if (nx >= 0 && nx < map.rows && ny >= 0 && ny < map.cols &&
                map.at<uchar>(nx, ny) == 255 && seen.at<uchar>(nx, ny) == 0)
            {
              pixels.push({nx, ny});
              seen.at<uchar>(nx, ny) = 1; // Mark as visited
            }
          };

          seen.ptr<uchar>(x)[y] = 1;

          // Check and push neighboring pixels (including diagonals).

          checkAndPush(x + 1, y);
          checkAndPush(x - 1, y);
          checkAndPush(x, y + 1);
          checkAndPush(x, y - 1);

          checkAndPush(x + 1, y + 1);
          checkAndPush(x - 1, y - 1);
          checkAndPush(x - 1, y + 1);
          checkAndPush(x + 1, y - 1);
        }
        regionIndex++;
      }
    }
  }
  return regionIndex - 1;
}

// Generate a random color.
Vec3b randomColor()
{
  return Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

/**
   Creates a colored visualization of the region map.
   @param Mat regionMap - a mat of type 8UC1 with each pixel's region id
   @return Mat - 8UC3 Mat with colors assigned to each region.
*/
Mat regionColor(Mat &regionMap)
{
  Mat coloredMap = Mat::zeros(regionMap.size(), CV_8UC3);
  map<int, Vec3b> colorTable;
  colorTable[0] = Vec3b(0, 0, 0);
  for (int i = 0; i < regionMap.rows; i++)
  {
    for (int j = 0; j < regionMap.cols; j++)
    {
      uchar regionIndex = regionMap.at<uchar>(i, j);
      if (colorTable.find(regionIndex) == colorTable.end())
      {
        colorTable[regionIndex] = randomColor();
      }
      // Set the pixel in the colored map to the color corresponding to its region
      coloredMap.at<Vec3b>(i, j) = colorTable[regionIndex];
    }
  }

  return coloredMap;
}

/**
    Removes regions smaller than minSize number of pixels.
    @param Mat regionMap - a mat of type 8UC1 with each pixel's region id.
    @param int minSize - the minimum pixel size to be considered a region.
    @return Mat - 8UC3 Mat with colors assigned to each region.
*/
Mat removeSmallRegions(Mat &regionMap, int minSize)
{
  Mat newRegionMap = Mat::zeros(regionMap.size(), CV_8UC1);
  map<int, int> regionSize;
  int numRegions = 0;
  for (int i = 0; i < regionMap.rows; i++)
  {
    for (int j = 0; j < regionMap.cols; j++)
    {
      int regionId = regionMap.ptr<uchar>(i)[j];
      regionSize[regionId] += 1;
    }
  }
  map<int, int> newIds;
  newIds[0] = 0;
  int currentId = 1;
  for (int i = 0; i < regionMap.rows; i++)
  {
    for (int j = 0; j < regionMap.cols; j++)
    {
      int regionId = regionMap.ptr<uchar>(i)[j];
      if (regionSize[regionId] >= minSize && regionMap.ptr<uchar>(i)[j] != 0)
      {
        if (newIds.find(regionId) == newIds.end())
        {
          newIds[regionId] = currentId++;
        }
        newRegionMap.ptr<uchar>(i)[j] = newIds[regionId];
      }
    }
  }
  return newRegionMap;
}

/**
 * Saves the feature vector to a file with the specified label and filename.
 *
 * @param features the region features to be saved
 * @param label the label for the feature vector
 * @param filename the name of the file to which the feature vector will be saved
 *
 * @return true if the feature vector is successfully saved, false otherwise
 *
 * @throws None
 */
bool saveFeatureVectorToFile(const RegionFeatures &features, const std::string &label, const std::string &filename)
{
  std::ofstream file(filename, std::ios::out | std::ios::app); // Open in append mode

  if (!file.is_open())
  {
    std::cerr << "Unable to open the file: " << filename << std::endl;
    return false;
  }

  // Pre-check for NaN or Inf values in features
  if (std::isnan(features.percentFilled) || std::isinf(features.percentFilled) ||
      std::isnan(features.boundingBoxAspectRatio) || std::isinf(features.boundingBoxAspectRatio))
  {
    std::cerr << "Invalid feature values for label: " << label << std::endl;
    return false;
  }

  // Write the features
  file << label << ","
       << features.percentFilled << ","
       << features.boundingBoxAspectRatio << ","
       << features.centroid.x << ","
       << features.centroid.y << ","
       << features.mainAxisMoment << ","
       << features.secondAxisMoment ;

  file << std::endl;
  file.close();
  return true;
}

/**
 * Rotates a 2D point around the origin by a given angle.
 *
 * @param p the point to rotate
 * @param theta the angle of rotation in radians
 *
 * @return the rotated point
 *
 * @throws None
 */
Coordinate rotatePoint(Coordinate &p, float theta)
{
  return {
      p.x * cos(theta) - p.y * sin(theta),
      p.x * sin(theta) + p.y * cos(theta)};
}


/**
 * Calculate the central moment of the given pixels around the specified centroid.
 *
 * @param pixels the vector of coordinates representing the pixels
 * @param centroidX the x-coordinate of the centroid
 * @param centroidY the y-coordinate of the centroid
 * @param orderMomentX the order of the moment in the x-direction
 * @param orederMonentY the order of the moment in the y-direction
 *
 * @return a tuple containing the calculated central moments (moment20, moment02, moment11)
 */
tuple<float, float, float> calculateCentralMoment(vector<Coordinate> pixels, float centroidX, float centroidY, int orderMomentX, int orederMonentY)
{
  float moment02 = 0;
  float moment20 = 0;
  float moment11 = 0;
  for (int i = 0; i < pixels.size(); i++)
  {
    Coordinate pixel = pixels[i];
    float differenceX = pixel.x - centroidX;
    float differenceY = pixel.y - centroidY;
    moment02 += differenceY * differenceY;
    moment20 += differenceX * differenceX;
    moment11 += differenceX * differenceY;
  }
  return make_tuple(moment20, moment02, moment11);
}

/**
 * Calculate the bounding box for the given list of coordinates.
 *
 * @param pixels reference to a vector of Coordinate objects
 *
 * @return AABB object representing the bounding box
 *
 * @throws None
 */
AABB boundingBox(vector<Coordinate> &pixels)
{
  AABB box;
  box.max.x = numeric_limits<float>::lowest(); // Use lowest possible value for max initialization
  box.max.y = numeric_limits<float>::lowest();
  box.min.x = numeric_limits<float>::max(); // Use max value for min initialization
  box.min.y = numeric_limits<float>::max();
  for (auto &pixel : pixels)
  {
    // Correctly update the bounding box coordinates
    if (pixel.x > box.max.x)
      box.max.x = pixel.x;
    if (pixel.y > box.max.y)
      box.max.y = pixel.y;
    if (pixel.x < box.min.x)
      box.min.x = pixel.x;
    if (pixel.y < box.min.y)
      box.min.y = pixel.y;
  }
  // cout << "aabb" << "\n";
  // cout << box.max.x << " " << box.max.y << "\n";
  // cout << box.min.x << " " << box.min.y << "\n";
  return box;
}

/**
 * Find the axis-aligned bounding box (AABB) for a set of pixels after applying rotation and translation.
 *
 * @param pixels a vector of Coordinate objects representing the pixels
 * @param orientation the angle of rotation in radians
 * @param centroidX the x-coordinate of the centroid
 * @param centroidY the y-coordinate of the centroid
 *
 * @return the axis-aligned bounding box (AABB) for the rotated and translated pixels
 *
 * @throws None
 */
AABB findAABB(vector<Coordinate> &pixels, float orientation, float centroidX, float centroidY)
{
  vector<Coordinate> rotatedPoints;
  for (auto &pixel : pixels)
  {
    Coordinate translated = {pixel.x - centroidX, pixel.y - centroidY};
    rotatedPoints.push_back(rotatePoint(translated, -orientation));
  }
  AABB aabb = boundingBox(rotatedPoints);
  return aabb;
}

/**
 * Calculate the oriented bounding box of a given region on the map.
 *
 * @param regionMap the input region map
 * @param targetID the target ID for the region
 * @param orientation the orientation angle
 * @param centroidX the x-coordinate of the centroid
 * @param centroidY the y-coordinate of the centroid
 *
 * @return the list of coordinates representing the oriented bounding box
 *
 * @throws ErrorType description of error
 */
vector<Coordinate> calculateOrientedBoundingBox(Mat &regionMap, int targetID, double orientation, float centroidX, float centroidY)
{
  vector<Coordinate> pixels;
  // centroidX = 0;
  // centroidY = 0;
  int count = 0;
  for (int i = 0; i < regionMap.rows; i++)
  {
    for (int j = 0; j < regionMap.cols; j++)
    {
      int regionId = regionMap.ptr<uchar>(i)[j];
      if (regionId == targetID)
      {
        Coordinate c = {static_cast<double>(j), static_cast<double>(i)};
        pixels.push_back(c);
        // centroidX += j;
        // centroidY += i;
        // count++;
      }
    }
  }
  // cout << "LOL: " << orientation << " " << centroidX << " " << centroidY;
  // centroidX /= count;
  // centroidY /= count;
  AABB aabb = findAABB(pixels, orientation, centroidX, centroidY);

  vector<Coordinate> corners = {
      {aabb.min.x, aabb.min.y},
      {aabb.max.x, aabb.min.y},
      {aabb.max.x, aabb.max.y},
      {aabb.min.x, aabb.max.y}};

  vector<Coordinate> obb;
  for (Coordinate &corner : corners)
  {
    // cout << corner.x << " " << corner.y;
    Coordinate rotatedBack = rotatePoint(corner, orientation);                            // Rotate back
    Coordinate originalPosition = {rotatedBack.x + centroidX, rotatedBack.y + centroidY}; // Translate back
    obb.push_back(originalPosition);
  }
  return obb;
}

/**
 * Calculate the filled percentage of a box.
 *
 * @param aabb the axis-aligned bounding box
 * @param pixelCount the count of filled pixels
 *
 * @return the filled percentage
 *
 * @throws None
 */
float boxFilledPercentage(AABB &aabb, int pixelCount)
{
  // cout << aabb.max.x << " " << aabb.max.y << " " << aabb.min.x << " " << aabb.min.y;
  float area = (aabb.max.x - aabb.min.x) * (aabb.max.y - aabb.min.y);
  return (float)pixelCount / area * 100.0;
}

/**
 * Computes the inertia of an object at a given angle.
 *
 * @param theta the angle at which to compute the inertia
 * @param moment20 the second moment around the x-axis
 * @param moment02 the second moment around the y-axis
 * @param moment11 the product moment around the x and y axes
 *
 * @return a tuple containing the computed inertia values (u20, u02)
 *
 * @throws None
 */
tuple<float, float> computeInertia(float theta, float moment20, float moment02, float moment11)
{
  float u20 = moment20 * cos(theta) * cos(theta) + moment02 * sin(theta) * sin(theta) + moment11 * sin(2 * theta);
  float u02 = moment20 * sin(theta) * sin(theta) + moment02 * cos(theta) * cos(theta) - moment11 * sin(2 * theta);
  return make_tuple(u20, u02);
}

/**
 * Compute region features from the given region map and target ID.
 *
 * @param regionMap the input region map
 * @param targetID the ID of the target region
 *
 * @return the computed region features
 *
 * @throws ErrorType description of error
 */
RegionFeatures computeRegionFeatures(const cv::Mat &regionMap, int targetID)
{
  float centroidX = 0;
  float centroidY = 0;
  int count = 0;
  vector<Coordinate> regionPixels;
  for (int i = 0; i < regionMap.rows; i++)
  {
    for (int j = 0; j < regionMap.cols; j++)
    {
      int regionId = regionMap.ptr<uchar>(i)[j];
      if (regionId == targetID)
      {
        Coordinate c = {(double)j, (double)i};
        regionPixels.push_back(c);
        centroidX += j;
        centroidY += i;
        count++;
      }
    }
  }

  if (count != 0)
  {
    centroidX /= count;
    centroidY /= count;
  }

  // Calculate the area of the region (number of non-zero pixels in the mask)
  float area = count;

  float m20, m02, m11;
  tie(m20, m02, m11) = calculateCentralMoment(regionPixels, centroidX, centroidY, 0, 0);

  float theta_radians = 0.5 * atan2(2 * m11, m20 - m02);
  float theta_degrees = theta_radians * (180.0 / M_PI);

  // cout << " rad " << theta_radians << "\n";
  AABB aabb = findAABB(regionPixels, theta_radians, centroidX, centroidY);

  float percentFilled = boxFilledPercentage(aabb, regionPixels.size());
  float heightWidthRatio = (aabb.max.x - aabb.min.x) / (aabb.max.y - aabb.min.y);

  float u20, u02;
  tie(u20, u02) = computeInertia(theta_radians, m20, m02, m11);

  // cout << "perc filled " << percentFilled;
  Point2f centroid(centroidX, centroidY);

  return RegionFeatures{(float)percentFilled, (float)heightWidthRatio, Point2f(centroidX, centroidY), theta_radians, u20, u02, area};
}

/**
 * A function to draw an axis on an image.
 *
 * @param image the image on which to draw the axis
 * @param theta the angle of the axis
 * @param centroidX the x-coordinate of the centroid
 * @param centroidY the y-coordinate of the centroid
 *
 * @return 0 on successful axis drawing
 *
 * @throws None
 */
int drawAxis(Mat &image, double theta, int centroidX, int centroidY)
{
  double L = 300;

  // Calculate axis endpoints
  Point pt1(centroidX + L / 2 * cos(theta), centroidY + L / 2 * sin(theta));
  Point pt2(centroidX - L / 2 * cos(theta), centroidY - L / 2 * sin(theta));

  // Draw the line on the image
  arrowedLine(image, pt1, pt2, cv::Scalar(0, 0, 255), 2);
  return 0;
}

/**
 * Draws an oriented bounding box (OBB) on the given image using the provided coordinates.
 *
 * @param image the image on which the OBB will be drawn
 * @param obb a vector of Coordinate objects representing the OBB points
 *
 * @return 0 on success, -1 if OBB does not contain exactly 4 points
 *
 * @throws None
 */
int drawObb(Mat &image, vector<Coordinate> obb)
{
  if (obb.size() != 4)
  {
    cerr << "Error: OBB must contain exactly 4 points." << endl;
    return -1;
  }

  Point a(obb[0].x, obb[0].y);
  Point b(obb[1].x, obb[1].y);
  Point c(obb[2].x, obb[2].y);
  Point d(obb[3].x, obb[3].y);
  // cout << "obb" << "\n";
  // cout << obb[0].x << " " << obb[0].y << "\n";
  // cout << obb[1].x << " " << obb[1].y << "\n";
  // cout << obb[2].x << " " << obb[2].y << "\n";
  // cout << obb[3].x << " " << obb[3].y << "\n";
  // Draw the line on the image
  line(image, a, b, Scalar(255, 0, 0), 2);
  line(image, b, c, cv::Scalar(255, 0, 0), 2);
  line(image, c, d, cv::Scalar(255, 0, 0), 2);
  line(image, d, a, cv::Scalar(255, 0, 0), 2);
  return 0;
}

/**
 * Calculate the Euclidean distance between two vectors.
 *
 * @param vec1 The first vector
 * @param vec2 The second vector
 *
 * @return The calculated Euclidean distance
 *
 * @throws None
 */
float calculateEuclideanDistance(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
  float distance = 0.0;
  for (size_t i = 0; i < vec1.size(); i++)
  {
    distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  return sqrt(distance);
}

/**
 * Calculates the scaled Euclidean distance between two vectors using the standard deviation for scaling.
 *
 * @param vec1 The first vector
 * @param vec2 The second vector
 * @param stdDev The standard deviation for scaling
 *
 * @return The scaled Euclidean distance between vec1 and vec2
 *
 * @throws None
 */
float calculateScaledEuclideanDistance(const std::vector<float> &vec1, const std::vector<float> &vec2, const std::vector<float> &stdDev)
{
  float distance = 0.0;
  for (size_t i = 0; i < vec1.size(); i++)
  {
    float scale = (stdDev[i] != 0) ? stdDev[i] : 1.0f; // Avoid division by zero
    distance += std::pow((vec1[i] - vec2[i]) / scale, 2);
  }
  return std::sqrt(distance);
}

/**
 * Compares the given features with the entries in the database and returns the label of the entry with the closest features.
 *
 * @param database The map of database entries
 * @param features The vector of features to compare
 * @param stdDev The vector of standard deviations for the features
 * @param minDistance The reference to the minimum distance found
 *
 * @return The label of the entry with the closest features
 *
 * @throws None
 */
std::string compareWithDatabase(const std::map<int, DatabaseEntry> &database, const std::vector<float> &features, const std::vector<float> &stdDev, float &minDistance)
{
  std::string closestLabel = "unknown";
  minDistance = std::numeric_limits<float>::max();

  for (const auto &entry : database)
  {
    float distance = calculateScaledEuclideanDistance(entry.second.features, features, stdDev);
    if (distance < minDistance)
    {
      minDistance = distance;
      closestLabel = entry.second.label;
      cout << "min distance: " << minDistance << "\n";
      cout << "closest label: " << closestLabel << "\n";
    }
  }

  return closestLabel;
}

/**
 * Loads the database entries from the given file.
 *
 * @param filename the name of the file to load
 *
 * @return a map containing the loaded database entries
 *
 * @throws N/A
 */
std::map<int, DatabaseEntry> loadDatabase(const std::string &filename)
{
  std::map<int, DatabaseEntry> database;
  std::ifstream file(filename);
  std::string line;
  int id = 0;

  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    std::string label;
    std::getline(iss, label, ',');

    DatabaseEntry entry;
    entry.label = label;

    float feature;
    while (iss >> feature)
    {
      entry.features.push_back(feature);
      if (iss.peek() == ',')
        iss.ignore();
    }
    
        database[id++] = entry;
  }

  return database;
}


/**
 * Calculates a distance based k-nearest neighbors and returns the object's label.
 * @param features - the object's features.
 * @param objects - object's and their labels from the database.
 * @param int numNeighbors - the number of neighbors to use.
 * @param float std - standart deviation of features.
 * @param minDistance - the minimal distance to the objects.
 * 
*/
String knn(const std::vector<float>& features, map<int, DatabaseEntry> &objects, int numNeighbors, vector<float> &std, float &minDistance){
      vector<pair<String, double>> distances;
      for(const auto& object : objects){
        distances.push_back(make_pair(object.second.label, calculateScaledEuclideanDistance(features, object.second.features, std)));
      }
      sort(distances.begin(), distances.end());

      map<String, vector<double>> closestClasses;
      for(pair<String, double> pair : distances){
        if(closestClasses[pair.first].size() < numNeighbors){
          closestClasses[pair.first].push_back(pair.second);
        }
      }
      
      map<String, double> closestClass;
      String label;
      double closestDistance = std::numeric_limits<double>::max();
      for (const auto& pair : closestClasses) {
        double distance = accumulate(pair.second.begin(), pair.second.end(), 0.0) / pair.second.size();
        if( distance < closestDistance){
          label = pair.first;
          closestDistance = distance;
        }
      }
      minDistance = closestDistance;
      return label;
    }

/**
 * Loads feature vectors from a file.
 *
 * @param filename the name of the file to load feature vectors from
 *
 * @return a vector of vectors of floats representing the feature vectors
 *
 * @throws std::invalid_argument if the file cannot be opened
 */
std::vector<std::vector<float>> loadFeatureVectors(const std::string &filename)
{
  std::vector<std::vector<float>> featureVectors;
  std::ifstream file(filename);
  std::string line;

  while (getline(file, line))
  {
    std::vector<float> featureVector;
    std::stringstream ss(line);
    std::string value;
    getline(ss, value, ','); // Skip the label

    while (getline(ss, value, ','))
    {
      featureVector.push_back(std::stof(value));
    }

    featureVectors.push_back(featureVector);
  }

  return featureVectors;
}


/**
 * Detects and labels regions in the given image using the provided region map and database file.
 *
 * @param image the input image
 * @param regionMap the map of regions in the image
 * @param databaseFilename the filename of the database containing feature vectors
 * @param method optional parameter indicating the method to use for labeling regions
 *
 * @return the label of the detected region
 *
 * @throws ErrorType description of error
 */
string detectAndLabelRegions(cv::Mat &image, const cv::Mat &regionMap, const std::string &databaseFilename, bool method = true)
{
  auto database = loadDatabase(databaseFilename);
  auto featureVectors = loadFeatureVectors(databaseFilename); // Assuming features are in the same file
  auto stdDev = calculateStandardDeviations(featureVectors);


  // Print the standard deviations
  // std::cout << "Standard Deviations:" << std::endl;
  // for (size_t i = 0; i < stdDev.size(); ++i) {
  //     std::cout << "Feature " << i + 1 << ": " << stdDev[i] << std::endl;
  // }

  double minVal, maxVal;
  cv::minMaxLoc(regionMap, &minVal, &maxVal);
  int maxRegionID = static_cast<int>(maxVal);
  std::string label;

  for (int regionID = 1; regionID <= maxRegionID; ++regionID)
  {
    RegionFeatures features = computeRegionFeatures(regionMap, regionID); 

    std::vector<float> featureVector = {
        features.percentFilled,
        features.boundingBoxAspectRatio,
        features.centroid.x,
        features.centroid.y,
        features.mainAxisMoment,
        features.secondAxisMoment,
        };

    float minDistance;
    if(method){
      cout << "scaled euclidean" << endl;
      label = compareWithDatabase(database, featureVector, stdDev, minDistance);
    }else{
      cout << "knn" << endl;
      label = knn(featureVector, database, 2, stdDev, minDistance);

    }
    //std::string label = compareWithDatabase(database, featureVector, stdDev, minDistance);
    //cout << "knn";
    
    // cout << "label: " << label << endl;
    cout << "minDistance: " << minDistance << endl;

    cv::Point labelPos(features.centroid.x, features.centroid.y);
    if (minDistance <= 0.75) {
        cv::putText(image, "Object: " + label , labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    } else {
        cv::putText(image, "unknown", labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    

    
  }
  return label;
}

/**
 * Calculate standard deviations for each feature in the given feature vectors.
 *
 * @param featureVectors the input feature vectors
 *
 * @return vector of standard deviations for each feature
 *
 * @throws None
 */
std::vector<float> calculateStandardDeviations(const std::vector<std::vector<float>> &featureVectors)
{
  std::vector<float> means(featureVectors[0].size(), 0.0f);
  for (const auto &vector : featureVectors)
  {
    for (size_t i = 0; i < vector.size(); ++i)
    {
      means[i] += vector[i];
    }
  }
  for (float &mean : means)
  {
    mean /= featureVectors.size();
  }


  std::vector<float> variances(means.size(), 0.0f);
  for (const auto& vector : featureVectors) {
      for (size_t i = 0; i < vector.size(); ++i) {
          variances[i] += (vector[i] - means[i]) * (vector[i] - means[i]);
      }
  }
  for (float& variance : variances) {
      variance /= featureVectors.size();
  }
  

  std::vector<float> standardDeviations;
  for (const float variance : variances)
  {
    standardDeviations.push_back(std::sqrt(variance));
  }

  return standardDeviations;
}

/**
 * Draws features on the input image based on the region name, region features, and the orientation of the bounding box.
 *
 * @param image the input image to draw features on
 * @param regionName the name of the region
 * @param features the features of the region
 * @param obb the oriented bounding box coordinates
 *
 * @return 0 upon successful completion
 */
int drawFeatures(Mat & image, String regionName, RegionFeatures features, vector<Coordinate> obb){
      int posy, posx = INT_MAX; 
      int maxY = 0;
      for(const Coordinate a : obb){
        if(a.y < posy){
          posy = a.y;
        }
        if (a.x < posx){
          posx = a.x;
        }
        if(a.y > maxY){
          maxY = a.y;
        }
      }
      int font = FONT_HERSHEY_SIMPLEX;
      float fontScale = 0.5;
      Scalar color(0, 255, 0);
      int thickness = 2;
      int lineType = LINE_AA;
      String text = "Percent Filled: " + to_string(features.percentFilled);
      
      posy = features.centroid.y - (maxY - posy) / 2;
      //Point position(features.centroid.x - 10, posy);
      Point position(features.centroid.x, features.centroid.y - 50);
      putText(image, text, position, font, fontScale, color, thickness, lineType);
      return 0;
    }

/**
 * Initializes the confusion matrix with zeros for all label combinations.
 *
 * @param confusionMatrix the confusion matrix to be initialized
 * @param labels the list of labels
 */
void initializeConfusionMatrix(map<string, map<string, int>>& confusionMatrix, const vector<string>& labels) {
    for (const auto& trueLabel : labels) {
        for (const auto& predictedLabel : labels) {
            confusionMatrix[trueLabel][predictedLabel] = 0;
        }
    }
}

/**
 * Update the confusion matrix with the given true and predicted labels.
 *
 * @param confusionMatrix the confusion matrix to be updated
 * @param trueLabel the true label
 * @param predictedLabel the predicted label
 *
 * @return void
 *
 * @throws None
 */
void updateConfusionMatrix(map<string, map<string, int>>& confusionMatrix, const string& trueLabel, const string& predictedLabel) {
    confusionMatrix[trueLabel][predictedLabel]++;
}

void printConfusionMatrix(const std::map<std::string, std::map<std::string, int>>& confusionMatrix) {
    // First, print the header row with labels
    cout << "\t";
    for (const auto& labelRow : confusionMatrix) {
        cout << labelRow.first << "\t";
    }
    cout << endl;

    // Now print each row of the confusion matrix
    for (const auto& labelRow : confusionMatrix) {
        // Print the row label
        cout << labelRow.first << "\t";
        for (const auto& labelColumn : labelRow.second) {
            // Print each cell in the row
            cout << labelColumn.second << "\t";
        }
        // End the row with a new line
        cout << endl;
    }
}

/**
 * Calculate the accuracy based on the confusion matrix.
 *
 * @param confusionMatrix the confusion matrix containing the true and predicted labels
 *
 * @return the accuracy as a double value
 *
 * @throws None
 */
double calculateAccuracy(const std::map<std::string, std::map<std::string, int>>& confusionMatrix) {
    int correctPredictions = 0;
    int totalPredictions = 0;

    for (const auto& trueLabelPair : confusionMatrix) {
        const std::string& trueLabel = trueLabelPair.first;
        for (const auto& predictedLabelPair : trueLabelPair.second) {
            const std::string& predictedLabel = predictedLabelPair.first;
            int count = predictedLabelPair.second;
            totalPredictions += count;
            if (trueLabel == predictedLabel) {
                correctPredictions += count;
            }
        }
    }

    return totalPredictions > 0 ? static_cast<double>(correctPredictions) / totalPredictions : 0;
}

/**
 * Retrieves the labels for classification.
 *
 * @return a vector of strings containing the labels
 */
std::vector<std::string> getLabels() {
    // You should populate this list with the labels you are classifying
    return {"controller", "gripper", "watch", "wallet", "vape"};
}
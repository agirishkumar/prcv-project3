#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "helpers.h"
#include <vector>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <stack>
#include <fstream>
#include <limits>

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

      if (pixelValue < thresh)
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
int regionGrowing(Mat &map, Mat &regionMap){

    stack<tuple<int, int>> pixels;
    Mat seen = Mat::zeros(map.size(), CV_8UC1);
    regionMap = Mat::zeros(map.size(), CV_8UC1);
    int regionIndex = 1;
    // Iterate throught the image.
    for(int i = 0; i < map.rows; i++){
        for(int j = 0; j < map.cols; j++){
            // If a pixel is foreground then push it to the stack.
            if(map.ptr<uchar>(i)[j] == 255 && regionMap.ptr<uchar>(i)[j] == 0){
                tuple<int, int> pixel = make_tuple(i, j);
                pixels.push(pixel);
                // pop pixel and assign current region, for each pixels neighbor check if it is a foreground and push it to stack
                while(!pixels.empty()){
                    auto [x, y] = pixels.top();
                    pixels.pop();

                    regionMap.ptr<uchar>(x)[y] = regionIndex;

                    auto checkAndPush = [&](int nx, int ny) {
                        if (nx >= 0 && nx < map.rows && ny >= 0 && ny < map.cols && 
                            map.at<uchar>(nx, ny) == 255 && seen.at<uchar>(nx, ny) == 0) {
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
Vec3b randomColor() {
    return Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

/**
   Creates a colored visualization of the region map.
   @param Mat regionMap - a mat of type 8UC1 with each pixel's region id
   @return Mat - 8UC3 Mat with colors assigned to each region.
*/
Mat regionColor(Mat &regionMap){
    Mat coloredMap = Mat::zeros(regionMap.size(), CV_8UC3);
    map<int, Vec3b> colorTable;
    colorTable[0] = Vec3b(0,0,0);
     for (int i = 0; i < regionMap.rows; i++) {
        for (int j = 0; j < regionMap.cols; j++) {
            uchar regionIndex = regionMap.at<uchar>(i, j);
            if (colorTable.find(regionIndex) == colorTable.end()) {
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
Mat removeSmallRegions(Mat &regionMap, int minSize){
    Mat newRegionMap = Mat::zeros(regionMap.size(), CV_8UC1);
    map<int, int> regionSize;
    int numRegions = 0;
    for (int i = 0; i < regionMap.rows; i++) {
        for (int j = 0; j < regionMap.cols; j++) {
            int regionId = regionMap.ptr<uchar>(i)[j];
            regionSize[regionId] += 1;
            
        }
    }
    map<int, int> newIds;
    newIds[0] = 0;
    int currentId = 1;
    for (int i = 0; i < regionMap.rows; i++) {
        for (int j = 0; j < regionMap.cols; j++) {
            int regionId = regionMap.ptr<uchar>(i)[j];
            if(regionSize[regionId] >= minSize && regionMap.ptr<uchar>(i)[j] != 0){
                if (newIds.find(regionId) == newIds.end()) {
                    newIds[regionId] = currentId++;
                }
                newRegionMap.ptr<uchar>(i)[j] = newIds[regionId];
            }
        }
    }
    return newRegionMap;
    
}

// Function to compute a color histogram for a region
cv::Mat calculateColorHistogram(const cv::Mat& image, const cv::Mat& mask, int bins = 256) {
    cv::Mat histogram;
    const int channels[] = {0, 1, 2}; // For a 3-channel image
    const int histSize[] = {bins, bins, bins};
    const float range[] = {0, 256}; // Pixel value range
    const float* ranges[] = {range, range, range};
    
    // Compute the histogram
    cv::calcHist(&image, 1, channels, mask, histogram, 3, histSize, ranges, true, false);

    // Normalize the histogram so that it's not affected by the image size
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX);

    return histogram;
}


/**
 * Computes the features of a region in the given region map.
 *
 * @param regionMap the input region map
 * @param regionID the ID of the region to compute features for
 *
 * @return a struct containing the percent filled and bounding box aspect ratio of the region
 *
 * @throws None
 */
RegionFeatures computeRegionFeatures(const cv::Mat& regionMap, int regionID, const cv::Mat& originalImage) {
    // Extract the region corresponding to the regionID to create a mask
    cv::Mat mask = regionMap == regionID;

    // Calculate the bounding box of the region
    cv::Rect boundingBox = cv::boundingRect(mask);

    // Calculate the area of the region (number of non-zero pixels in the mask)
    float area = cv::countNonZero(mask);

    // Calculate the percent filled: ratio of the region area to the bounding box area
    float percentFilled = area / static_cast<float>(boundingBox.area());

    // Calculate the aspect ratio of the bounding box
    float boundingBoxAspectRatio = static_cast<float>(boundingBox.height) / boundingBox.width;

    // Calculate the centroid of the region
    cv::Moments m = cv::moments(mask, true);
    cv::Point2f centroid(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));

    // Calculate the color histogram for the region using the mask
    cv::Mat histogram = calculateColorHistogram(originalImage, mask);

    // Return the calculated features, including the histogram
    return {percentFilled, boundingBoxAspectRatio, centroid, histogram};
}






/**
 * Display region features on the given image.
 *
 * @param image the input image
 * @param regionMap the region map
 * @param regionID the ID of the region to display
 * @param features the features of the region to display
 *
 * @return void
 *
 * @throws None
 */
// void displayRegionFeatures(Mat& image, const Mat& regionMap, int regionID, const RegionFeatures& features) {

//     // Mat invertedRegionMap;
//     // bitwise_not(regionMap, invertedRegionMap);

//     Mat region = regionMap == regionID;
//     Rect boundingBox = boundingRect(region);
//     rectangle(image, boundingBox, Scalar(0, 255, 0), 2);
//     string text = "Fill: " + to_string(features.percentFilled) + ", Aspect: " + to_string(features.boundingBoxAspectRatio);
//     putText(image, text, boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
// }

// void displayRegionFeatures(Mat& image, const Mat& regionMap, int regionID, const RegionFeatures& features) {
//     Mat region = regionMap == regionID;
//     Rect boundingBox = boundingRect(region);

//     // Create the text to display. Include the region ID.
//     string text = "ID: " + to_string(regionID) + 
//                   " Fill: " + to_string(features.percentFilled) + 
//                   ", Aspect: " + to_string(features.boundingBoxAspectRatio);

//     // Define the bottom-left corner of the text based on the bounding box
//     Point textOrg(boundingBox.x, boundingBox.y + boundingBox.height + 20); // Move the text below the bounding box

//     // Draw the bounding box and put the text
//     rectangle(image, boundingBox, Scalar(0, 255, 0), 2);
//     putText(image, text, textOrg, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
// }


bool saveFeatureVectorToFile(const RegionFeatures& features, const std::string& label, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::app); // Open in append mode
    
    if (!file.is_open()) {
        std::cerr << "Unable to open the file: " << filename << std::endl;
        return false;
    }
    
    // Write the basic features
    file << label << ","
         << features.percentFilled << ","
         << features.boundingBoxAspectRatio << ","
         << features.centroid.x << ","
         << features.centroid.y;
    
    // Flatten the histogram to a single row and write it to the file
    if (!features.colorHistogram.empty()) {
        cv::Mat flatHistogram;
        if (features.colorHistogram.isContinuous()) {
            flatHistogram = features.colorHistogram.reshape(1, 1); // Flatten the histogram
        } else {
            // Clone the data into a continuous Mat before reshaping if it's not already continuous
            flatHistogram = features.colorHistogram.clone().reshape(1, 1);
        }

        // Write each histogram bin to the file
        for (int i = 0; i < flatHistogram.cols; i++) {
            file << "," << flatHistogram.at<float>(0, i);
        }
    }
    
    file << std::endl; // End of line for this feature vector
    file.close(); // Close the file
    return true;
}


Coordinate rotatePoint(Coordinate &p, double theta){
        return {
            p.x * cos(theta) - p.y * sin(theta),
            p.x * sin(theta) + p.y * cos(theta)
        };
    }

//calculates central moments.
    tuple<float, float, float> calculateCentralMoment(vector<Coordinate> pixels, float centroidX, float centroidY, int orderMomentX, int orederMonentY){
        float moment02 = 0;
        float moment20 = 0;
        float moment11 = 0;
        for(int i = 0; i < pixels.size(); i++){
            Coordinate pixel = pixels[i];
            float differenceX = pixel.x - centroidX;
            float differenceY = pixel.y - centroidY;
            moment02 +=  differenceY *  differenceY;
            moment20 += differenceX * differenceX;
            moment11 += differenceX * differenceY;
        }
        return make_tuple(moment20, moment02, moment11);
    }

    AABB boundingBox(vector<Coordinate> &pixels){
        AABB box;
        box.max.x = numeric_limits<double>::lowest(); // Use lowest possible value for max initialization
        box.max.y = numeric_limits<double>::lowest();
        box.min.x = numeric_limits<double>::max(); // Use max value for min initialization
        box.min.y = numeric_limits<double>::max();
        for (auto &pixel : pixels) {
            // Correctly update the bounding box coordinates
            if (pixel.x > box.max.x) box.max.x = pixel.x;
            if (pixel.y > box.max.y) box.max.y = pixel.y;
            if (pixel.x < box.min.x) box.min.x = pixel.x; 
            if (pixel.y < box.min.y) box.min.y = pixel.y;
        }
        cout << "aabb" << "\n";
        cout << box.max.x << " " << box.max.y << "\n";
        cout << box.min.x << " " << box.min.y << "\n";
        return box;
    }

    //Finds axis aligned bounding box
    AABB findAABB(vector<Coordinate> &pixels, double orientation, float centroidX, float centroidY){
        vector<Coordinate> rotatedPoints;
        for(auto &pixel : pixels){
            Coordinate translated = {pixel.x - centroidX, pixel.y - centroidY};
            rotatedPoints.push_back(rotatePoint(translated, -orientation));
        }
        AABB aabb = boundingBox(rotatedPoints);
        return aabb;
    }
    // finds the oriented bounding box
    vector<Coordinate> calculateOrientedBoundingBox(Mat &regionMap, int targetID, double orientation, float centroidX, float centroidY){
        vector<Coordinate> pixels;
        // centroidX = 0;
        // centroidY = 0;
        int count = 0;
        for (int i = 0; i < regionMap.rows; i++) {
            for (int j = 0; j < regionMap.cols; j++) {
                int regionId = regionMap.ptr<uchar>(i)[j];
                if(regionId == targetID){
                    Coordinate c = {static_cast<double>(j), static_cast<double>(i)};
                    pixels.push_back(c);
                    // centroidX += j;
                    // centroidY += i;
                    // count++;
                }
            }
            
        }
        cout << "LOL: " << orientation << " " << centroidX << " " << centroidY;
        // centroidX /= count;
        // centroidY /= count;

        AABB aabb = findAABB(pixels, orientation, centroidX, centroidY);

        vector<Coordinate> corners = {
            {aabb.min.x, aabb.min.y},
            {aabb.max.x, aabb.min.y},
            {aabb.max.x, aabb.max.y},
            {aabb.min.x, aabb.max.y}
        };
        
        vector<Coordinate> obb;
        for (Coordinate& corner : corners) {
            cout << corner.x << " " << corner.y;
            Coordinate rotatedBack = rotatePoint(corner, orientation); // Rotate back
            Coordinate originalPosition = {rotatedBack.x + centroidX, rotatedBack.y + centroidY}; // Translate back
            obb.push_back(originalPosition);
        }
        return obb;
    }

    
    double boxFilledPercentage(AABB &aabb, int pixelCount){
        cout << aabb.max.x << " " << aabb.max.y << " " << aabb.min.x << " " << aabb.min.y;
        double area = (aabb.max.x - aabb.min.x) * (aabb.max.y - aabb.min.y);
        return (double)pixelCount / area * 100.0;
    }

    tuple<double, double> computeInertia(double theta, double moment20, double moment02, double moment11){
        double u20 = moment20 * cos(theta) * cos(theta) + moment02 * sin(theta) * sin(theta) + moment11 * sin(2*theta);
        double u02 = moment20 * sin(theta) * sin(theta) + moment02 * cos(theta) * cos(theta) - moment11 * sin(2*theta);
        return make_tuple(u20, u02);
    }

    RegionFeatures computeRegionFeatures(Mat &regionMap, int targetID){
        float centroidX = 0;
        float centroidY = 0;
        int count = 0;
        vector<Coordinate> regionPixels;
        for (int i = 0; i < regionMap.rows; i++) {
            for (int j = 0; j < regionMap.cols; j++) {
                int regionId = regionMap.ptr<uchar>(i)[j];
                if(regionId == targetID){
                    Coordinate c = {(double)j, (double)i};
                    regionPixels.push_back(c);
                    centroidX += j;
                    centroidY += i;
                    count++;
                }
            }
        }
        if (count != 0){
            centroidX /= count;
            centroidY /= count;
        }

        float m20, m02, m11;
        tie(m20, m02, m11) = calculateCentralMoment(regionPixels, centroidX, centroidY, 0, 0);

        double theta_radians = 0.5 * atan2(2*m11, m20 - m02);
        double theta_degrees = theta_radians * (180.0 / M_PI);

        cout << " rad " << theta_radians << "\n";
        AABB aabb = findAABB(regionPixels, theta_radians, centroidX, centroidY);

        double percentFilled = boxFilledPercentage(aabb, regionPixels.size());
        double heightWidthRatio = (aabb.max.x - aabb.min.x) / (aabb.max.y - aabb.min.y);

        double u20, u02;
        tie(u20, u02) = computeInertia(theta_radians, m20, m02, m11);


        cout << "perc filled " << percentFilled;
        Point2f centroid(centroidX, centroidY);

        // TODO: chage histogram to actual
        return {(float)percentFilled, (float)heightWidthRatio, centroid, regionMap, theta_radians, u20, u02};
    }


    //draws the principal axis.
    int drawAxis(Mat &image, double theta, int centroidX, int centroidY){
        double L = 100; 

        // Calculate axis endpoints
        Point pt1(centroidX + L/2 * cos(theta), centroidY + L/2 * sin(theta));
        Point pt2(centroidX - L/2 * cos(theta), centroidY - L/2 * sin(theta));

        // Draw the line on the image
        line(image, pt1, pt2, cv::Scalar(0, 0, 255), 2);
        return 0;
    }

    // draws the oriented Bounding box
    int drawObb(Mat &image, vector<Coordinate> obb){
        if (obb.size() != 4) {
            cerr << "Error: OBB must contain exactly 4 points." << endl;
            return -1; 
        }

        Point a(obb[0].x, obb[0].y); 
        Point b(obb[1].x, obb[1].y); 
        Point c(obb[2].x, obb[2].y);
        Point d(obb[3].x, obb[3].y);
        cout << "obb" << "\n";
        cout << obb[0].x << " " << obb[0].y << "\n";
        cout << obb[1].x << " " << obb[1].y << "\n";
        cout << obb[2].x << " " << obb[2].y << "\n";
        cout << obb[3].x << " " << obb[3].y << "\n";
        // Draw the line on the image
        line(image, a, b, Scalar(255, 0, 0), 2);
        line(image, b, c, cv::Scalar(255, 0, 0), 2);
        line(image, c, d, cv::Scalar(255, 0, 0), 2);
        line(image, d, a, cv::Scalar(255, 0, 0), 2);
        return 0;
    }

    






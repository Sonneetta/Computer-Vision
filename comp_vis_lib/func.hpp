#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace comp_vis {

    Mat histogramEqualization(const Mat& src);
    vector<int> computeHistogram(const Mat& src);
    void displayHistogram(const std::vector<int>& hist, const std::string& filename);

    Mat myblur(const Mat& input, int kernelSize);
    Mat local_operator(const Mat& input, const Mat& kernel);

    Mat sharpenImage(const Mat& inputImage, const Mat& blurredImage, double alpha);

    void computeGradients(const Mat& input, Mat& gradX, Mat& gradY);
    std::vector<cv::Point> fastDetector(const cv::Mat& image, int threshold);

    void customBoxFilter(const Mat& src, Mat& dst, Size kernelSize);
    void customMedianFilter(const Mat& src, Mat& dst, int kernelSize);
    void customGaussianFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma);
    void customSigmaFilter(const Mat& src, Mat& dst, int diameter, double sigmaColor, double sigmaSpace);
    void computeGradients(const Mat& input, Mat& gradX, Mat& gradY);
    void harrisCornerDetector(const Mat& input, Mat& output, double k = 0.04);
}
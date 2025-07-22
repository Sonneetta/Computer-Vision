#include "func.hpp"
#include <cstdint>
#include <opencv2/opencv.hpp>

namespace comp_vis

{
    std::vector<int> computeHistogram(const Mat& src) {

        int height = src.rows;
        int width = src.cols;
        int histSize = 256;

        std::vector<int> hist(histSize, 0);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int pixelValue = src.at<uchar>(i, j);
                hist[pixelValue]++;
            }
        }

        return hist;
    }


    std::vector<float> normalizeHistogram(const std::vector<int>& hist) {

        int histSize = hist.size();
        std::vector<float> cdf(histSize, 0);

        float totalPixels = 0;
        for (int i = 0; i < histSize; ++i) {
            totalPixels += hist[i];
            cdf[i] = totalPixels;
        }


        for (int i = 0; i < histSize; ++i) {
            cdf[i] = cdf[i] / totalPixels;
        }

        return cdf;
    }

    Mat histogramEqualization(const Mat& src) {
        Mat dst = src.clone();


        std::vector<int> hist = computeHistogram(src);


        std::vector<float> cdf = normalizeHistogram(hist);


        for (int i = 0; i < dst.rows; ++i) {
            for (int j = 0; j < dst.cols; ++j) {
                int pixelValue = src.at<uchar>(i, j);
                dst.at<uchar>(i, j) = static_cast<uchar>(cdf[pixelValue] * 255);
            }
        }


        return dst;
    }

    void displayHistogram(const std::vector<int>& hist, const std::string& title) {

        int histHeight = 400;
        int histWidth = 512;
        int binWidth = cvRound((double)histWidth / hist.size());

        Mat histogramImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));


        for (size_t i = 0; i < hist.size(); i++) {

            int normalizedHeight = cvRound((double)hist[i] * histHeight / *std::max_element(hist.begin(), hist.end()));
            rectangle(histogramImage, Point(i * binWidth, histHeight),
                Point((i + 1) * binWidth, histHeight - normalizedHeight),
                Scalar(0, 0, 255), -1);
        }

        imshow(title, histogramImage);

    }

    Mat myblur(const Mat& input, int kernelSize) {
        Mat output = input.clone();

        if (kernelSize % 2 == 0) {
            cout << "Kernel size must be odd." << endl;
            return output;
        }

        int offset = kernelSize / 2;

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                double sum = 0.0;
                int count = 0;

                for (int j = -offset; j <= offset; j++) {
                    for (int i = -offset; i <= offset; i++) {
                        int newY = min(max(y + j, 0), input.rows - 1);
                        int newX = min(max(x + i, 0), input.cols - 1);
                        sum += input.at<uchar>(newY, newX);
                        count++;
                    }
                }

                output.at<uchar>(y, x) = saturate_cast<uchar>(sum / count);
            }
        }

        return output;
    }

    Mat local_operator(const Mat& input, const Mat& kernel) {
        Mat output = input.clone();

        int kernelSize = kernel.rows;
        if (kernelSize % 2 == 0) {
            cout << "Kernel size must be odd." << endl;
            return output;
        }

        int offset = kernelSize / 2;

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                double sum = 0.0;

                for (int j = -offset; j <= offset; j++) {
                    for (int i = -offset; i <= offset; i++) {
                        int newY = min(max(y + j, 0), input.rows - 1);
                        int newX = min(max(x + i, 0), input.cols - 1);
                        sum += input.at<uchar>(newY, newX) * kernel.at<double>(j + offset, i + offset);
                    }
                }

                output.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            }
        }

        return output;
    }

    Mat sharpenImage(const Mat& inputImage, const Mat& blurredImage, double alpha) {

        Mat highPassImage = inputImage - blurredImage;
        Mat outputImage = inputImage + alpha * highPassImage;
        return outputImage;

    }

    void customNormalize(const Mat& input, Mat& output, double newMin, double newMax) {
        double minVal, maxVal;
        minMaxLoc(input, &minVal, &maxVal); 

        output = Mat::zeros(input.size(), input.type());

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
             
                double normalizedValue = ((input.at<double>(y, x) - minVal) / (maxVal - minVal)) * (newMax - newMin) + newMin;
                output.at<double>(y, x) = saturate_cast<uchar>(normalizedValue);
            }
        }
    }

    void computeGradients(const Mat& input, Mat& gradX, Mat& gradY) { 
        
        double sobelX[3][3] = { {-1, 0, 1},
                                {-2, 0, 2},
                                {-1, 0, 1} };

        double sobelY[3][3] = { {-1, -2, -1},
                                {0, 0, 0},
                                {1, 2, 1} };

       
        int rows = input.rows;
        int cols = input.cols;
        gradX = Mat::zeros(rows, cols, CV_64F);
        gradY = Mat::zeros(rows, cols, CV_64F);

        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                double sumX = 0.0;
                double sumY = 0.0;

                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        sumX += input.at<uchar>(y + j, x + i) * sobelX[j + 1][i + 1];
                        sumY += input.at<uchar>(y + j, x + i) * sobelY[j + 1][i + 1];
                    }
                }

                gradX.at<double>(y, x) = sumX;
                gradY.at<double>(y, x) = sumY;
            }
        }
    }

    void harrisCornerDetector(const Mat& input, Mat& output, double k) {
        Mat gradX, gradY;
        computeGradients(input, gradX, gradY);

     
        Mat Ixx = gradX.mul(gradX);
        Mat Iyy = gradY.mul(gradY);
        Mat Ixy = gradX.mul(gradY);

        int rows = input.rows;
        int cols = input.cols;
        output = Mat::zeros(rows, cols, CV_64F);

       
        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                double sumIxx = 0.0, sumIyy = 0.0, sumIxy = 0.0;

              
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        sumIxx += Ixx.at<double>(y + j, x + i);
                        sumIyy += Iyy.at<double>(y + j, x + i);
                        sumIxy += Ixy.at<double>(y + j, x + i);
                    }
                }

             
                double det = sumIxx * sumIyy - sumIxy * sumIxy;
                double trace = sumIxx + sumIyy;
                output.at<double>(y, x) = det - k * trace * trace;
            }
        }

       
        Mat normalizedOutput;
        customNormalize(output, normalizedOutput, 0, 255);
        normalizedOutput.convertTo(output, CV_8U);
    }

    std::vector<cv::Point> fastDetector(const cv::Mat& image, int threshold) {
        std::vector<cv::Point> keypoints;
        int offset[16][2] = {
            {0, 3}, {1, 3}, {2, 2}, {3, 1}, {3, 0}, {3, -1}, {2, -2}, {1, -3},
            {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
        };

        for (int y = 3; y < image.rows - 3; ++y) {
            for (int x = 3; x < image.cols - 3; ++x) {
                int centerPixel = image.at<uchar>(y, x);
                int countBright = 0, countDark = 0;

                for (int i = 0; i < 16; ++i) {
                    int newY = y + offset[i][0];
                    int newX = x + offset[i][1];
                    int pixel = image.at<uchar>(newY, newX);

                    if (pixel > centerPixel + threshold) {
                        countBright++;
                    }
                    else if (pixel < centerPixel - threshold) {
                        countDark++;
                    }

                    if (countBright >= 12 || countDark >= 12) {

                        bool isUnique = true;
                        for (const auto& kp : keypoints) {
                            if (cv::norm(kp - cv::Point(x, y)) < 5) {
                                isUnique = false;
                                break;
                            }
                        }
                        if (isUnique) {
                            keypoints.push_back(cv::Point(x, y));
                        }
                        break;
                    }
                }
            }
        }
        return keypoints;
    }

    void customBoxFilter(const Mat& src, Mat& dst, Size kernelSize)
    {
        dst = Mat::zeros(src.size(), src.type());
        int kx = kernelSize.width / 2;
        int ky = kernelSize.height / 2;
        for (int y = ky; y < src.rows - ky; y++)
        {
            for (int x = kx; x < src.cols - kx; x++)
            {
                float sum = 0.0;
                for (int j = -ky; j <= ky; j++)
                {
                    for (int i = -kx; i <= kx; i++)
                    {
                        sum += src.at<uchar>(y + j, x + i);
                    }
                }
                dst.at<uchar>(y, x) = sum / (kernelSize.width * kernelSize.height);
            }
        }
    }

    void customMedianFilter(const Mat& src, Mat& dst, int kernelSize)
    {
        dst = Mat::zeros(src.size(), src.type());
        int k = kernelSize / 2;
        for (int y = k; y < src.rows - k; y++)
        {
            for (int x = k; x < src.cols - k; x++)
            {
                std::vector<uchar> neighborhood;
                for (int j = -k; j <= k; j++)
                {
                    for (int i = -k; i <= k; i++)
                    {
                        neighborhood.push_back(src.at<uchar>(y + j, x + i));
                    }
                }
                sort(neighborhood.begin(), neighborhood.end());
                dst.at<uchar>(y, x) = neighborhood[neighborhood.size() / 2];
            }
        }
    }

    void customGaussianFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma)
    {
        dst = Mat::zeros(src.size(), src.type());
        int kx = kernelSize.width / 2;
        int ky = kernelSize.height / 2;
        std::vector<std::vector<double>> kernel(kernelSize.height, vector<double>(kernelSize.width));
        double sum = 0.0;


        for (int y = -ky; y <= ky; y++)
        {
            for (int x = -kx; x <= kx; x++)
            {
                kernel[y + ky][x + kx] = exp(-(x * x + y * y) / (2 * sigma * sigma));
                sum += kernel[y + ky][x + kx];
            }
        }

        for (int y = 0; y < kernelSize.height; y++)
        {
            for (int x = 0; x < kernelSize.width; x++)
            {
                kernel[y][x] /= sum;
            }
        }


        for (int y = ky; y < src.rows - ky; y++)
        {
            for (int x = kx; x < src.cols - kx; x++)
            {
                float sum = 0.0;
                for (int j = -ky; j <= ky; j++)
                {
                    for (int i = -kx; i <= kx; i++)
                    {
                        sum += src.at<uchar>(y + j, x + i) * kernel[j + ky][i + kx];
                    }
                }
                dst.at<uchar>(y, x) = sum;
            }
        }
    }

    void customSigmaFilter(const Mat& src, Mat& dst, int diameter, double sigmaColor, double sigmaSpace)
    {
        dst = Mat::zeros(src.size(), src.type());
        int radius = diameter / 2;
        for (int y = radius; y < src.rows - radius; y++)
        {
            for (int x = radius; x < src.cols - radius; x++)
            {
                float sum = 0.0;
                float weightSum = 0.0;
                uchar center = src.at<uchar>(y, x);
                for (int j = -radius; j <= radius; j++)
                {
                    for (int i = -radius; i <= radius; i++)
                    {
                        uchar neighbor = src.at<uchar>(y + j, x + i);
                        double spaceWeight = exp(-(i * i + j * j) / (2 * sigmaSpace * sigmaSpace));
                        double colorWeight = exp(-((neighbor - center) * (neighbor - center)) /
                            (2 * sigmaColor * sigmaColor));
                        double weight = spaceWeight * colorWeight;
                        sum += neighbor * weight;
                        weightSum += weight;
                    }
                }
                dst.at<uchar>(y, x) = sum / weightSum;
            }
        }
    }


} 



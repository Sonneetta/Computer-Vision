#include "func.hpp"
#include <cstdint>
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc.hpp> 
#include <set>

namespace comp_vis {

    Point2f calc_centroid(const Mat& image) {
        float sumX = 0, sumY = 0;
        int count = 0;

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                if (image.at<uchar>(y, x) == 255) {
                    sumX += x;
                    sumY += y;
                    count++;
                }
            }
        }

        return Point2f(sumX / count, sumY / count);
    }

    void visualize_centroid(Mat& image, const Point2f& centroid) {
        if (centroid.x >= 0 && centroid.y >= 0) {
            circle(image, Point(static_cast<int>(centroid.x), static_cast<int>(centroid.y)), 4, Scalar(0, 0, 255), -1);
        }
    }

    void calc_major_axis(const Mat& image, const Point2f& centroid, Point2f& axis_start, Point2f& axis_end, float& eccentricity) {
        float sumXX = 0, sumYY = 0, sumXY = 0;
        int count = 0;

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                if (image.at<uchar>(y, x) == 255) {
                    sumXX += (x - centroid.x) * (x - centroid.x);
                    sumYY += (y - centroid.y) * (y - centroid.y);
                    sumXY += (x - centroid.x) * (y - centroid.y);
                    count++;
                }
            }
        }

        if (count == 0) {
            axis_start = Point2f(0, 0);
            axis_end = Point2f(0, 0);
            eccentricity = 0;
            return;
        }

        float covXX = sumXX / count;
        float covYY = sumYY / count;
        float covXY = sumXY / count;

        float trace = covXX + covYY;
        float determinant = covXX * covYY - covXY * covXY;

        float eigenValue1 = trace / 2 + sqrt(trace * trace / 4 - determinant);
        float eigenValue2 = trace / 2 - sqrt(trace * trace / 4 - determinant);

        float a = sqrt(eigenValue1);
        float b = sqrt(eigenValue2);

        eccentricity = sqrt(1 - (b * b) / (a * a));

        float theta = 0.5 * atan2(2 * covXY, covXX - covYY);
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        float length = 150;
        axis_start = centroid + Point2f(-length * cosTheta, -length * sinTheta);
        axis_end = centroid + Point2f(length * cosTheta, length * sinTheta);
    }

    void visualize_major_axis(Mat& image, const Point2f& axis_start, const Point2f& axis_end) {
        line(image, Point(static_cast<int>(axis_start.x), static_cast<int>(axis_start.y)),
            Point(static_cast<int>(axis_end.x), static_cast<int>(axis_end.y)),
            Scalar(255, 0, 0), 2);
    }

    void cannyEdgeDetection(const Mat& src, Mat& dst, double lowerThreshold, double upperThreshold) {
        dst = Mat::zeros(src.size(), CV_8U);
        Mat gradientMagnitude = Mat::zeros(src.size(), CV_64F);
        Mat gradientDirection = Mat::zeros(src.size(), CV_64F);

        int sobelKernelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int sobelKernelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        for (int y = 1; y < src.rows - 1; y++) {
            for (int x = 1; x < src.cols - 1; x++) {
                double gx = 0.0, gy = 0.0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        gx += sobelKernelX[i + 1][j + 1] * src.at<uchar>(y + i, x + j);
                        gy += sobelKernelY[i + 1][j + 1] * src.at<uchar>(y + i, x + j);
                    }
                }
                gradientMagnitude.at<double>(y, x) = sqrt(gx * gx + gy * gy);
                gradientDirection.at<double>(y, x) = atan2(gy, gx);
            }
        }

        Mat nonMaxSuppressed = Mat::zeros(src.size(), CV_64F);
        for (int y = 1; y < src.rows - 1; y++) {
            for (int x = 1; x < src.cols - 1; x++) {
                double direction = gradientDirection.at<double>(y, x) * 180.0 / CV_PI;
                direction = fmod(direction + 180.0, 180.0);

                double magnitude = gradientMagnitude.at<double>(y, x);
                double q = 0.0, r = 0.0;

                if ((0 <= direction && direction < 22.5) || (157.5 <= direction && direction <= 180)) {
                    q = gradientMagnitude.at<double>(y, x + 1);
                    r = gradientMagnitude.at<double>(y, x - 1);
                }
                else if (22.5 <= direction && direction < 67.5) {
                    q = gradientMagnitude.at<double>(y + 1, x - 1);
                    r = gradientMagnitude.at<double>(y - 1, x + 1);
                }
                else if (67.5 <= direction && direction < 112.5) {
                    q = gradientMagnitude.at<double>(y + 1, x);
                    r = gradientMagnitude.at<double>(y - 1, x);
                }
                else if (112.5 <= direction && direction < 157.5) {
                    q = gradientMagnitude.at<double>(y - 1, x - 1);
                    r = gradientMagnitude.at<double>(y + 1, x + 1);
                }

                if (magnitude >= q && magnitude >= r) {
                    nonMaxSuppressed.at<double>(y, x) = magnitude;
                }
            }
        }

        for (int y = 1; y < src.rows - 1; y++) {
            for (int x = 1; x < src.cols - 1; x++) {
                if (nonMaxSuppressed.at<double>(y, x) >= upperThreshold) {
                    dst.at<uchar>(y, x) = 255;
                }
                else if (nonMaxSuppressed.at<double>(y, x) < lowerThreshold) {
                    dst.at<uchar>(y, x) = 0;
                }
                else {
                    bool connected = false;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            if (dst.at<uchar>(y + i, x + j) == 255) {
                                connected = true;
                                break;
                            }
                        }
                        if (connected) break;
                    }
                    dst.at<uchar>(y, x) = connected ? 255 : 0;
                }
            }
        }

    }

    void houghTransform(const Mat& edges, vector<pair<float, float>>& lines, int threshold) {
        int width = edges.cols;
        int height = edges.rows;
        int maxRho = sqrt(width * width + height * height);
        int numAngles = 180;
        vector<vector<int>> accumulator(2 * maxRho, vector<int>(numAngles, 0));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                if (edges.at<uchar>(y, x) == 255) {

                    for (int theta = 0; theta < numAngles; theta++) {
                        float rad = theta * CV_PI / 180.0;
                        int rho = cvRound(x * cos(rad) + y * sin(rad)) + maxRho;
                        if (rho >= 0 && rho < 2 * maxRho) {
                            accumulator[rho][theta]++;
                        }
                    }
                }
            }
        }

        for (int rho = 0; rho < 2 * maxRho; rho++) {
            for (int theta = 0; theta < numAngles; theta++) {
                if (accumulator[rho][theta] >= threshold) {
                    lines.emplace_back(rho - maxRho, theta * CV_PI / 180.0);
                }
            }
        }
    }

    void drawHoughLines(Mat& img, const vector<pair<float, float>>& lines) {
        for (const auto& line : lines) {
            float rho = line.first;
            float theta = line.second;
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            cv::line(img, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
        }
    }

    double curvature(Point p1, Point p2, Point p3) {
        double dx1 = p2.x - p1.x;
        double dy1 = p2.y - p1.y;
        double dx2 = p3.x - p2.x;
        double dy2 = p3.y - p2.y;

        double num = abs(dx1 * dy2 - dy1 * dx2);
        double denom = pow(dx1 * dx1 + dy1 * dy1, 1.5);

        if (denom == 0) return 0;

        return num / denom;
    }

    void calculateCurvature(const vector<vector<Point>>& contours, vector<Point>& maxCurvaturePoints, vector<Point>& minCurvaturePoints,
        double& maxCurvature, double& minCurvature, double& avgCurvature, int& pointCount) {

        const double epsilon = 1e-5;
        double sumCurvature = 0;
        maxCurvature = -std::numeric_limits<double>::infinity();
        minCurvature = std::numeric_limits<double>::infinity();
        pointCount = 0;

        for (const auto& contour : contours) {
            for (size_t i = 0; i < contour.size(); ++i) {
                Point p1 = contour[(i == 0) ? contour.size() - 1 : i - 1]; 
                Point p2 = contour[i];
                Point p3 = contour[(i == contour.size() - 1) ? 0 : i + 1]; 
                double curv = curvature(p1, p2, p3);
                sumCurvature += curv;
                pointCount++;


                if (curv > maxCurvature) {
                    maxCurvature = curv;
                    maxCurvaturePoints.clear();
                    maxCurvaturePoints.push_back(contour[i]);
                }
                else if (curv == maxCurvature) {
                    maxCurvaturePoints.push_back(contour[i]);
                }


                if (curv < minCurvature - epsilon) {
                    minCurvature = curv;
                    minCurvaturePoints.clear();
                    minCurvaturePoints.push_back(contour[i]);
                }
                else if (abs(curv - minCurvature) < epsilon) {
                    minCurvaturePoints.push_back(contour[i]);
                }
            }
        }


        if (pointCount > 0) {
            avgCurvature = sumCurvature / pointCount;
        }
        else {
            avgCurvature = 0;
        }
    }


    Mat markCurvaturePoints(const Mat& contourImage, const vector<Point>& maxPoints, const vector<Point>& minPoints) {
        Mat markedImage = contourImage.clone();

        for (const auto& pt : minPoints) {
            circle(markedImage, pt, 2, Scalar(255, 0, 0), -1);
        }

        for (const auto& pt : maxPoints) {
            circle(markedImage, pt, 5, Scalar(0, 0, 255), -1);
        }

        return markedImage;
    }

    void outputMaxCurvaturePoints(const vector<Point>& maxCurvaturePoints) {
        cout << "Max Curvature Points:" << endl;
        for (const auto& pt : maxCurvaturePoints) {
            cout << "Point: (" << pt.x << ", " << pt.y << ")" << endl;
        }
    }

 

}

#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace comp_vis {

	Point2f calc_centroid(const Mat& image);
	void visualize_centroid(Mat& image, const Point2f& centroid);
	void calc_major_axis(const Mat& image, const Point2f& centroid, Point2f& axis_start, Point2f& axis_end, float& eccentricity);
	void visualize_major_axis(Mat& image, const Point2f& axis_start, const Point2f& axis_end);
	void cannyEdgeDetection(const Mat& src, Mat& dst, double lowerThreshold = 50, double upperThreshold = 150);
	void houghTransform(const Mat& edges, vector<pair<float, float>>& lines, int threshold);
	void drawHoughLines(Mat& img, const vector<pair<float, float>>& lines);
	double curvature(Point p1, Point p2, Point p3);
	void calculateCurvature(const vector<vector<Point>>& contours, vector<Point>& maxCurvaturePoints, vector<Point>& minCurvaturePoints,
	double& maxCurvature, double& minCurvature, double& avgCurvature, int& pointCount);
	Mat markCurvaturePoints(const Mat& contourImage, const vector<Point>& maxPoints, const vector<Point>& minPoints);
	void outputMaxCurvaturePoints(const vector<Point>& maxCurvaturePoints);




}
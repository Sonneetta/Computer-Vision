#include <opencv2/opencv.hpp>
#include "func.hpp"
#include "task.hpp"
#include <numeric>

using namespace cv;
using namespace std;

void task3(int subtask)
{
    Mat input = imread("D:\\Study\\4_curse_1_sem\\CV\\lab_2\\images\\input\\4.jpg", IMREAD_GRAYSCALE);
    if (input.empty()) {
        cout << "Could not read the image" << endl;
        return;
    }   
    Mat binary;
    threshold(input, binary, 128, 255, THRESH_BINARY);

    int width = binary.cols;
    int height = binary.rows;

    /*  cout << "Pixel values of the input binary image:" << endl;
      for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
              cout << static_cast<int>(binary.at<uchar>(y, x)) << " ";
          }
          cout << endl;
      }*/

    Mat output;
    cvtColor(binary, output, COLOR_GRAY2BGR);

    imshow("Task 3 Original image", binary);
    moveWindow("Task 3 Original image", 0, 0);

    switch (subtask) {
    case 1: {
        Point2f centroid = comp_vis::calc_centroid(binary);

        comp_vis::visualize_centroid(output, centroid);

        imshow("MY Centroid Visualization", output);
        moveWindow("MY Centroid Visualization", width, 0);

        Point2f axis_start, axis_end;
        float my_eccentricity;
        comp_vis::calc_major_axis(binary, centroid, axis_start, axis_end, my_eccentricity);
        comp_vis::visualize_major_axis(output, axis_start, axis_end);
        imshow("MY Major Axis Visualization", output);
        moveWindow("MY Major Axis Visualization", width, height); 

        // TASK 3 OPENCV
        Mat centr_cv = binary.clone();
        cvtColor(binary, centr_cv, COLOR_GRAY2BGR);
        Moments m = moments(binary, true);

        if (m.m00 == 0) {
            cout << "No white area found!" << endl;
            return;
        }

        Point p(m.m10 / m.m00, m.m01 / m.m00);
        circle(centr_cv, p, 4, Scalar(0, 0, 255), -1);
        imshow("OpenCV Centroid Visualization", centr_cv);
        moveWindow("OpenCV Centroid Visualization", 2 * width, 0);  

        Mat covMatrix = (Mat_<double>(2, 2) << m.mu20 / m.m00, m.mu11 / m.m00, m.mu11 / m.m00, m.mu02 / m.m00);

        Mat eigenValues, eigenVectors;
        eigen(covMatrix, eigenValues, eigenVectors);

        Point2f mainAxis(eigenVectors.at<double>(0, 0), eigenVectors.at<double>(0, 1));

        float halfLength = 150.0;
        Point2f startPoint = centroid - halfLength * mainAxis;
        Point2f endPoint = centroid + halfLength * mainAxis;

        line(centr_cv, startPoint, endPoint, Scalar(255, 0, 0), 2);

        imshow("OpenCV Main Axis Visualization", centr_cv);
        moveWindow("OpenCV Main Axis Visualization", 2 * width, height);  

        cout << "My Centroid: (" << static_cast<int>(centroid.x) << ", " << static_cast<int>(centroid.y) << ")" << endl;
        cout << "OpenCV Centroid: (" << p.x << ", " << p.y << ")" << endl;
        cout << "My major axis start: (" << axis_start.x << ", " << axis_start.y << "), end: (" << axis_end.x << ", " << axis_end.y << ")" << endl;
        cout << "OpenCV major axis start: (" << startPoint.x << ", " << startPoint.y << "), end: (" << endPoint.x << ", " << endPoint.y << ")" << endl;

        float a = sqrt(eigenValues.at<double>(0));
        float b = sqrt(eigenValues.at<double>(1));
        float eccentricity_cv = sqrt(1 - (b * b) / (a * a));
        cout << "OpenCV eccentricity: " << eccentricity_cv << endl;
        cout << "My eccentricity: " << my_eccentricity << endl;

        cout << "Press any key to return to menu..." << endl;
        waitKey(0);
        break;
    }
    default:
        cout << "Invalid subtask choice." << endl;
        break;
    }
}





void task5(int subtask) {

    Mat input = imread("D:\\Study\\4_curse_1_sem\\CV\\lab_2\\images\\input\\14.jpg", IMREAD_GRAYSCALE); 
    if (input.empty()) { 
        cout << "Could not read the image" << endl;
        return;
    }

    Mat binary;
    threshold(input, binary, 128, 255, THRESH_BINARY);

    imshow("Task 5 Original image", binary);
    moveWindow("Task 5 Original image", 0, 0);

    switch (subtask) {
    case 1: {

       //Task 5 
       vector<vector<Point>> contours;

        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        Mat contourImage = Mat::zeros(binary.size(), CV_8UC3);
        drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 1);

        imshow("Contour Image", contourImage);

        vector<Point> maxCurvaturePoints, minCurvaturePoints;
        double maxCurvature = 0, minCurvature = DBL_MAX; 
        double avgCurvature = 0;
        int pointCount = 0;

        comp_vis::calculateCurvature(contours, maxCurvaturePoints, minCurvaturePoints, maxCurvature, minCurvature, avgCurvature, pointCount);

        cout << "Maximum curvature: " << maxCurvature << endl;
        cout << "Minimum curvature: " << minCurvature << endl;
        cout << "Average curvature: " << avgCurvature << endl;


        comp_vis::outputMaxCurvaturePoints(maxCurvaturePoints);

        Mat markedImage = comp_vis::markCurvaturePoints(contourImage, maxCurvaturePoints, minCurvaturePoints);
       
        imshow("Marked Image with Curvature Points", markedImage);



        cout << "Press any key to return to menu..." << endl;
        waitKey(0);
        break;
    }
    default:
        cout << "Invalid subtask choice." << endl;
        break;
    }
}

void task7(int subtask) {

    Mat input = imread("D:\\Study\\4_curse_1_sem\\CV\\lab_2\\images\\input\\6.jpg", IMREAD_GRAYSCALE); 
    if (input.empty()) { 
        cout << "Could not read the image" << endl; 
        return;
    }

   
    Mat binary;
    threshold(input, binary, 128, 255, THRESH_BINARY);

    imshow("Task 7 Original image", binary);
    moveWindow("Task 7 Original image", 0, 0);

    switch (subtask) {
    case 1: {

        Mat my_edges;
        comp_vis::cannyEdgeDetection(binary, my_edges);
        imshow("My Canny Edges", my_edges);

        vector<pair<float, float>> my_lines;
        comp_vis::houghTransform(my_edges, my_lines, 70);
        Mat my_imgWithLines;
        cvtColor(binary, my_imgWithLines, COLOR_GRAY2BGR);
        comp_vis::drawHoughLines(my_imgWithLines, my_lines);
        imshow("My Detected Hough Lines", my_imgWithLines);


        // Task 7 OpenCV 
        Mat edges;
        Canny(binary, edges, 50, 150);
        imshow("OpenCV Canny Edges", edges);

        vector<Vec2f> lines;

        HoughLines(edges, lines, 1, CV_PI / 180, 60);

        Mat imgWithLines;
        cvtColor(binary, imgWithLines, COLOR_GRAY2BGR);

        for (size_t i = 0; i < lines.size(); i++) {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(imgWithLines, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
        }

        imshow("OpenCV Detected  Hough Lines", imgWithLines);
        moveWindow("OpenCV Detected  Hough Lines", 0, 0);

        cout << "Press any key to return to menu..." << endl;
        waitKey(0);
        break;
    }
    default:
        cout << "Invalid subtask choice." << endl;
        break;
    }
}
#include <opencv2/opencv.hpp>

#include "func.hpp"
#include "task.hpp"

using namespace cv;
using namespace std;

void task1(int subtask)
{
    Mat img = imread("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\input\\image.jpg", IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Could not read the image" << endl;
        return;
    }

    switch (subtask) {

    case 1: {
        vector<int> originalHist = comp_vis::computeHistogram(img);
        comp_vis::displayHistogram(originalHist, "Task 1 MY Original Histogram");

        Mat equalizedImg = comp_vis::histogramEqualization(img);
        vector<int> equalizedHist = comp_vis::computeHistogram(equalizedImg);
        comp_vis::displayHistogram(equalizedHist, "task 1 MY Equalized Histogram");

        cout << "Press any key to return to menu..." << endl;
        waitKey(0);
        break;
    }
    case 2: {
        Mat equalizedImg = comp_vis::histogramEqualization(img);

        imshow("Task 1 Original Image", img);
        moveWindow("Task 1 Original Image", 0, 0);

        imshow("Task 1 MY Equalized Image", equalizedImg);
        moveWindow("Task 1 MY Equalized Image", 500, 0);

        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_image_equalized.jpg", equalizedImg);

        Mat cv_equ_hist = img.clone();
        equalizeHist(img, cv_equ_hist);

        imshow("Task 1 OpenCV Equalized Image", cv_equ_hist);
        moveWindow("Task 1 OpenCV Equalized Image", 1000, 0);

        cout << "Press any key to return to menu..." << endl;
        waitKey(0);
        break;
    }
    default:
        cout << "Invalid subtask choice." << endl;
        break;
    }
}

void task3(int subtask)
{
    Mat img = imread("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\input\\image_2.jpg", IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Could not read the image" << endl;
        return;
    }

    Mat kernel_box = (Mat_<double>(5, 5) << 1, 1, 1, 1, 1, 
                                            1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1) / 25.0;

    Mat kernel_sobelX = (Mat_<double>(5, 5) <<  
        -2, -1, 0, 1, 2,
        -4, -3, 0, 3, 4,
        -6, -5, 0, 5, 6,
        -4, -3, 0, 3, 4,
        -2, -1, 0, 1, 2);
   /*  
    Mat kernel_sobelX = (Mat_<double>(5, 5) <<  
        -1, -2, 0, 2, 1,
        -4, -8, 0, 8, 4,
        -6, -12, 0, 12, 6,
        -4, -8, 0, 8, 4,
        -1, -2, 0, 2, 1);*/

    /*Mat kernel_sobelX = (Mat_<double>(5, 5) << 
        -1, -2, 0, 2, 1,
        -4, -6, 0, 6, 4,
        -6, -10, 0, 10, 6,
        -4, -6, 0, 6, 4,
        -1, -2, 0, 2, 1);*/


   /* Mat kernel_sobelX = (Mat_<double>(3, 3) << 
        -1, 0, 1, 
        -2, 0, 2,
        -1, 0, 1);*/

    switch (subtask) {
    case 1: {
        Mat image_box = comp_vis::local_operator(img, kernel_box);

        imshow("Task 3 Original Image", img);
        moveWindow("Task 3 Original Image", 0, 0);

        imshow("Task 3 MY local operator box Blurred Image", image_box);
        moveWindow("Task 3 MY local operator box Blurred Image", 500, 0);

        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_blurred_image.jpg", image_box);

        Mat result;
        blur(img, result, Size(5, 5));
        imshow("Task 3  OpenCV Blur Image", result);
        moveWindow("Task 3  OpenCV Blur Image", 1000, 0);

        waitKey(0);
        break;
    }
    case 2: {
        imshow("Task 3 Original Image", img);
        moveWindow("Task 3 Original Image", 0, 0);

        Mat image_sobelX = comp_vis::local_operator(img, kernel_sobelX);
        imshow("Task 3 MY local operator SobelX Image", image_sobelX);
        moveWindow("Task 3 MY local operator SobelX Image", 500, 0);

        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_sobelX_image.jpg", image_sobelX);

        Mat grad_x;
        Sobel(img, grad_x, CV_8U, 1, 0, 5);
        Mat abs_grad_x;
        convertScaleAbs(grad_x, abs_grad_x);
        imshow("Task 3 OpenCV SobelX Image", abs_grad_x);
        moveWindow("Task 3 OpenCV SobelX Image", 1000, 0);

        waitKey(0);
        break;
    }
    default:
        cout << "Invalid subtask choice." << endl; 
        break;
    }
}


void task5(int subtask)
{
    Mat img = imread("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\input\\image_31.jpg", IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Could not read the image" << endl;
        return;
    }

    double alpha = 1.5;
    Size kernelSize(5, 5);
    double sigma = 1.0;
    int sigmaFilterDiameter = 9;
    double sigmaColor = 75.0;
    double sigmaSpace = 75.0;

    Mat blurredImage, sharpenedImageBox, sharpenedImageMedian, sharpenedImageGaussian, sharpenedImageSigma;

    int startX = 20;
    int windowOffset = 500; 

    switch (subtask) {
    case 1:

        comp_vis::customBoxFilter(img, blurredImage, kernelSize);
        sharpenedImageBox = comp_vis::sharpenImage(img, blurredImage, alpha);
        imshow("Original Image", img);
        moveWindow("Original Image", startX, 20); 
        imshow("Blurred (Box Filter)", blurredImage);
        moveWindow("Blurred (Box Filter)", startX + windowOffset, 20); 
        imshow("Sharpened (Box Filter)", sharpenedImageBox);
        moveWindow("Sharpened (Box Filter)", startX + 2 * windowOffset, 20); 
        waitKey(0);
        break;

    case 2:

        comp_vis::customMedianFilter(img, blurredImage, kernelSize.width);
        sharpenedImageMedian = comp_vis::sharpenImage(img, blurredImage, alpha);
        imshow("Original Image", img);
        moveWindow("Original Image", startX, 20);
        imshow("Blurred (Median Filter)", blurredImage);
        moveWindow("Blurred (Median Filter)", startX + windowOffset, 20);
        imshow("Sharpened (Median Filter)", sharpenedImageMedian);
        moveWindow("Sharpened (Median Filter)", startX + 2 * windowOffset, 20);
        waitKey(0);
        break;

    case 3:

        comp_vis::customGaussianFilter(img, blurredImage, kernelSize, sigma);
        sharpenedImageGaussian = comp_vis::sharpenImage(img, blurredImage, alpha);
        imshow("Original Image", img);
        moveWindow("Original Image", startX, 20);
        imshow("Blurred (Gaussian Filter)", blurredImage);
        moveWindow("Blurred (Gaussian Filter)", startX + windowOffset, 20);
        imshow("Sharpened (Gaussian Filter)", sharpenedImageGaussian);
        moveWindow("Sharpened (Gaussian Filter)", startX + 2 * windowOffset, 20);
        waitKey(0);
        break;

    case 4:

        comp_vis::customSigmaFilter(img, blurredImage, sigmaFilterDiameter, sigmaColor, sigmaSpace);
        sharpenedImageSigma = comp_vis::sharpenImage(img, blurredImage, alpha);
        imshow("Original Image", img);
        moveWindow("Original Image", startX, 20);
        imshow("Blurred (Sigma Filter)", blurredImage);
        moveWindow("Blurred (Sigma Filter)", startX + windowOffset, 20);
        imshow("Sharpened (Sigma Filter)", sharpenedImageSigma);
        moveWindow("Sharpened (Sigma Filter)", startX + 2 * windowOffset, 20);
        waitKey(0);
        break;

    case 5:

        comp_vis::customBoxFilter(img, blurredImage, kernelSize);
        sharpenedImageBox = comp_vis::sharpenImage(img, blurredImage, alpha);

        comp_vis::customMedianFilter(img, blurredImage, kernelSize.width);
        sharpenedImageMedian = comp_vis::sharpenImage(img, blurredImage, alpha);

        comp_vis::customGaussianFilter(img, blurredImage, kernelSize, sigma);
        sharpenedImageGaussian = comp_vis::sharpenImage(img, blurredImage, alpha);

        comp_vis::customSigmaFilter(img, blurredImage, sigmaFilterDiameter, sigmaColor, sigmaSpace);
        sharpenedImageSigma = comp_vis::sharpenImage(img, blurredImage, alpha);

        imshow("Original Image", img);
        moveWindow("Original Image", startX, 20);

        imshow("Sharpened (Box Filter)", sharpenedImageBox);
        moveWindow("Sharpened (Box Filter)", startX + windowOffset, 20);

        imshow("Sharpened (Median Filter)", sharpenedImageMedian);
        moveWindow("Sharpened (Median Filter)", startX + 2 * windowOffset, 20);

        imshow("Sharpened (Gaussian Filter)", sharpenedImageGaussian);
        moveWindow("Sharpened (Gaussian Filter)", startX + 3 * windowOffset, 20);

        imshow("Sharpened (Sigma Filter)", sharpenedImageSigma);
        moveWindow("Sharpened (Sigma Filter)", 20, 500);

        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_sharpenedImageBox.jpg", sharpenedImageBox);
        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_sharpenedImageMedian.jpg", sharpenedImageMedian); 
        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_sharpenedImageGaussian.jpg", sharpenedImageGaussian);  
        imwrite("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\output\\my_sharpenedImageSigma.jpg", sharpenedImageSigma);

        waitKey(0);
        break;

    default:
        cout << "Invalid choice!" << endl;
        return;
    }
}


void task7(int subtask)
{
    switch (subtask) {
    case 1:
    {
        Mat image = imread("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\input\\image_74.jpg");
        if (image.empty()) {
            cout << "Could not read the image" << endl;
            return;
        }

        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        int blockSize = 2;
        int ksize = 3;
        double k = 0.04;
        double threshold = 100;

        Mat harrisCorners;
        cornerHarris(grayImage, harrisCorners, blockSize, ksize, k, BORDER_DEFAULT);

        Mat harrisCornersNorm;
        normalize(harrisCorners, harrisCornersNorm, 0, 255, NORM_MINMAX, CV_32FC1);

        Mat outputImage = image.clone();
        for (int y = 0; y < harrisCornersNorm.rows; y++) {
            for (int x = 0; x < harrisCornersNorm.cols; x++) {
                if (harrisCornersNorm.at<float>(y, x) > threshold) {

                    circle(outputImage, Point(x, y), 3, Scalar(0, 255, 0), 1);
                }
            }
        }

        imshow("Îrigin", image);
        imshow("Îpencv Harris Corners", outputImage);

        Mat myharrisCorners;
        comp_vis::harrisCornerDetector(image, harrisCorners);

        imshow("MY Harris Corners", harrisCorners); 

       

        waitKey(0); 
    }
    break;
    case 2:
    {
        Mat image = imread("D:\\Study\\4_curse_1_sem\\CV\\comp_vis\\images\\input\\image7.jpg",IMREAD_GRAYSCALE);

        if (image.empty()) {
            cout << "Could not read the image" << endl;
            return;
        }

        vector<KeyPoint> keypoints;
        int threshold = 40;


        Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold);
        fast->detect(image, keypoints);

        Mat outputImage;
        drawKeypoints(image, keypoints, outputImage, Scalar(0, 255, 0));

        imshow("ORIGINAL IMAGE", image);
        imshow("FAST Corners Opencv", outputImage);

        int threshold_1 = 40;

        vector<Point> corners = comp_vis::fastDetector(image, threshold_1);
        Mat outputImageWithCorners;
        cvtColor(image, outputImageWithCorners, COLOR_GRAY2BGR);

        for (const auto& point : corners) {
            circle(outputImageWithCorners, point, 4, Scalar(0, 255, 0), 1);
        }

        imshow("MY FAST Corners", outputImageWithCorners);
      
        waitKey(0);
    }
    break;
    default:
        cout << "Invalid subtask choice." << endl;
        break;
    }

}
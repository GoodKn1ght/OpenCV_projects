#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void main() {
    string path = "Resources/test.png";
    Mat img = imread(path);
    Mat imgGray;
    Mat imgBlur;
    Mat medianimg;
    Mat cannyimg1, cannyimg2;
    Mat imgDil, imgErode;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imgBlur, Size(7, 7), 100, 0);
    medianBlur(img, medianimg, 7);
    Canny(imgBlur, cannyimg1, 50, 150);
    Canny(medianimg, cannyimg2, 50, 50);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    dilate(cannyimg1, imgDil, kernel);
    erode(imgDil, imgErode, kernel);

    imshow("Image", img);
    imshow("Image Gray", imgGray);
    imshow("Image Blur", imgBlur);
    imshow("Median Blur", medianimg);
    imshow("Canny gausian Blur", cannyimg1);
    imshow("Canny median Blur", cannyimg2);
    imshow("Image Dilation", imgDil);
    imshow("Image Erode", imgErode);
    // my custom kernel 
    Mat custom_kernel = (Mat_<float>(3, 3) <<
        1, 2, 1,
        2, 3, 2,
        1, 2, 1
        );
    float normalization_factor = 24.0f; // makes it darker 
    custom_kernel = custom_kernel / normalization_factor;
    Mat output_img = imread(path);
    Mat output_image;
    filter2D(img,output_image, -1, custom_kernel);
    imshow("Filtered Image", output_image);
    waitKey(0);
}
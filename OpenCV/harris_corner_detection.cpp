#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void main() {
	string path = "Resources/paper.jpg";
	Mat img = imread(path);

	Mat imgGray, imgBlur, imgCorner;
	Mat imgCornerNormalized; 
	Mat imgCornerScaled; 

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(5, 5), 1, 1);
	cornerHarris(imgGray, imgCorner, 2, 3, 0.04);
	normalize(imgCorner, imgCornerNormalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(imgCornerNormalized, imgCornerScaled);
	resize(imgCornerScaled, imgCornerScaled, Size(), 0.5, 0.5);
	imshow("Original Image", img);
	imshow("Harris Response (Scaled for View)", imgCornerScaled);
	waitKey(0);
}
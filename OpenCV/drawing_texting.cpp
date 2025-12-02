#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void main() {
	Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));
	circle(img, Point(256, 256), 155, Scalar(0, 69, 255),FILLED);
	rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), FILLED);
	line(img, Point(130, 296), Point(382, 350), Scalar(255, 255, 255), 2);
	putText(img, "Write smth", Point(200, 262), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
	imshow("image", img);
	waitKey(0);
}
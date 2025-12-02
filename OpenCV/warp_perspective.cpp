#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void main() {
	string path = "Resources/cards.jpg";
	Mat img = imread(path);
	Mat matrix, imgWarp;
	Mat imgWarpQueen;
	float w = 250;
	float h = 350;
	Point2f src[4] = { {529, 142}, {771, 190}, {405, 395}, {674, 457} };
	Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0, h}, {w,h} };
	Point2f src_queen[4] = { {62, 322}, {336, 276}, {91, 637}, {402, 572} };
	matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w,h));

	matrix = getPerspectiveTransform(src_queen, dst);
	warpPerspective(img, imgWarpQueen, matrix, Point(w, h));
	imshow("queen Perspective", imgWarpQueen);
	imshow("king Perspective", imgWarp);
	matrix = getAffineTransform(src, dst);
	warpAffine(img, imgWarp, matrix, Point(w, h));
	for (int i = 0; i < 4; i++) {
		circle(img, Point(src[i]), 10, Scalar(0, 0, 255), FILLED);
	}
	for (int i = 0; i < 4; i++) {
		circle(img, Point(src_queen[i]), 10, Scalar(0, 0, 255), FILLED);
	}
	imshow("cards", img);
	imshow("king Affine", imgWarp);
	waitKey(0);
}
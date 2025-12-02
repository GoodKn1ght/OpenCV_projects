#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void drawBounding(Mat img, Rect& bbox, string& label) {
    rectangle(img, bbox.tl(), bbox.br(), Scalar(0, 255, 0), 2);
    Point textPos(bbox.x, bbox.y - 5);
    if (bbox.y < 15) {
        textPos = Point(bbox.x + 5, bbox.y + 15);
    }
    putText(img, label, textPos, FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 0, 0), 2, LINE_AA);
}

void countFigures(Mat img, int k) {
    Mat imgGray;
    resize(img, img, Size(), 0.4, 0.4);
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    medianBlur(imgGray, imgGray, 3);
    Mat imgThreshold;
    threshold(imgGray, imgThreshold, 127, 255, THRESH_BINARY_INV);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgThreshold, imgThreshold, MORPH_CLOSE, kernel);
    imshow("Threshold", imgThreshold);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgThreshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<double> areas(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        areas[i] = contourArea(contours[i]);
    }
    int triangles = 0;
    int circles = 0;
    int quadrangles = 0;
    int pentagons = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = areas[i];
        if (area > 100) {
            bool isValidChild = true;
            if (hierarchy[i][3] != -1) {
                int parentIdx = hierarchy[i][3];
                double parentArea = areas[parentIdx];
                double areaRatio = area / parentArea;
                if (areaRatio > 0.4) {
                    isValidChild = false;
                }
            }
            if (isValidChild) {
                double peri = arcLength(contours[i], true);
                approxPolyDP(contours[i], conPoly[i], 0.03 * peri, true);
                boundRect[i] = boundingRect(conPoly[i]);
                int cornerCount = conPoly[i].size();
                string objectType;
                if (cornerCount == 3) {
                    objectType = "Triangle";
                    triangles++;
                }
                else if (cornerCount == 4) {
                    objectType = "Quadrangle";
                    quadrangles++;
                }
                else if (cornerCount > 5) {
                    objectType = "Circle";
                    circles++;
                }
                else if (cornerCount == 5) {
                    objectType = "Pentagon";
                    pentagons++;
                }
                Scalar color;
                if (hierarchy[i][3] == -1) {
                    color = Scalar(255, 0, 255); 
                }
                else {
                    color = Scalar(0, 255, 255); 
                }
                drawContours(img, conPoly, i, color, 2);
                drawBounding(img, boundRect[i], objectType);
            }
        }
    }
    putText(img, "Triangles: " + to_string(triangles), Point(30, 30), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
    putText(img, "Circles: " + to_string(circles), Point(30, 50), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
    putText(img, "Quadrangles: " + to_string(quadrangles), Point(30, 70), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
    putText(img, "Pentagons: " + to_string(pentagons), Point(30, 90), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
    putText(img, "Photo: " + to_string(k), Point(30, 110), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 2);
    imshow("Result", img);
    waitKey(0);
}

int main() {
    string path_example = "Resources/test_*.jpg";
    for (int i = 0; i < 10; i++) {
        path_example[15] = i + '0';
        Mat img = imread(path_example);
        countFigures(img, i);
    }
    return 0;
}
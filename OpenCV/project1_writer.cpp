#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
struct ColorInfo {
    string name;
    Scalar lower_hsv;
    Scalar upper_hsv;
    Scalar bgr_color;
};
Mat plot;
Point prevPoint(-1, -1);
vector<ColorInfo> myColors = {
    {"Red", Scalar(0, 100, 100), Scalar(10, 255, 255), Scalar(0, 0, 255)},
    {"Red", Scalar(160, 100, 100), Scalar(179, 255, 255), Scalar(0, 0, 255)},
    {"Green", Scalar(35, 100, 100), Scalar(75, 255, 255), Scalar(0, 255, 0)},
    {"Blue", Scalar(95, 100, 100), Scalar(135, 255, 255), Scalar(255, 0, 0)},
    {"Yellow", Scalar(20, 100, 100), Scalar(30, 255, 255), Scalar(0, 255, 255)}
};
Point findColor(Mat imgHSV, Mat& outputFrame, Scalar& drawColor) {
    Point tipPoint(-1, -1);
    double maxArea = 0;
    string foundColorName = "";
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    int Y_ZERO_THRESHOLD = 100;
    for ( auto& color : myColors) {
        Mat mask, mask_processed;
        inRange(imgHSV, color.lower_hsv, color.upper_hsv, mask);
        erode(mask, mask_processed, kernel);
        dilate(mask_processed, mask_processed, kernel);
        vector<vector<Point>> contours;
        findContours(mask_processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& contour : contours) {
            double area = contourArea(contour);
            if (area > 1000) {
                if (area > maxArea) {
                    maxArea = area;
                    foundColorName = color.name;
                    drawColor = color.bgr_color;
                    Point actualMinY = contour[0];
                    Point actualMaxY = contour[0];
                    for (auto& p : contour) {
                        if (p.y < actualMinY.y) actualMinY = p;
                        if (p.y > actualMaxY.y) actualMaxY = p;
                    }
                    if (actualMinY.y < Y_ZERO_THRESHOLD) {
                        tipPoint = actualMaxY; 
                        circle(outputFrame, actualMaxY, 5, Scalar(0, 255, 255), FILLED); 
                        circle(outputFrame, actualMinY, 5, Scalar(255, 0, 255), 1);     

                    }
                    else {
                        tipPoint = actualMinY; 
                        circle(outputFrame, actualMinY, 5, Scalar(0, 255, 255), FILLED); 
                        circle(outputFrame, actualMaxY, 5, Scalar(255, 0, 255), 1);     
                    }
                    Rect bbox = boundingRect(contour);
                    rectangle(outputFrame, bbox, drawColor, 2);
                }
            }
        }
    }

    if (maxArea > 0) {
        circle(outputFrame, tipPoint, 10, drawColor, FILLED);
        putText(outputFrame, foundColorName, Point(tipPoint.x + 20, tipPoint.y - 10), FONT_HERSHEY_DUPLEX, 0.7, drawColor, 2);
    }
    return tipPoint;
}
void drawOnPlot(Point currentPoint, Scalar drawColor) {
    if (currentPoint.x != -1 && currentPoint.y != -1) {
        if (prevPoint.x == -1) {
            prevPoint = currentPoint;
        }
        else {
            line(plot, prevPoint, currentPoint, drawColor, 10, LINE_AA);
            prevPoint = currentPoint;
        }
    }
    else {
        prevPoint = Point(-1, -1);
    }
}
void main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "ERROR: Cannot open webcam." << endl;
        return;
    }
    Mat img, imgHSV;
    Scalar drawColor(0, 0, 0);
    cap.read(img);
    if (img.empty()) return;
    plot = Mat::zeros(img.size(), CV_8UC3);
    while (true) {
        cap.read(img);
        if (img.empty()) break;
        cvtColor(img, imgHSV, COLOR_BGR2HSV);
        Point currentPoint = findColor(imgHSV, img, drawColor);
        drawOnPlot(currentPoint, drawColor);
        addWeighted(img, 1.0, plot, 1.0, 0.0, img);
        imshow("Image", img);
        if (waitKey(1) == 27) {
            break;
        }
    }
}
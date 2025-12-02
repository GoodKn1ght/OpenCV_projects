#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm> 
using namespace cv;
using namespace std;

vector<Point2f> reorder(vector<Point> points) {
    vector<Point2f> newPoints(4);
    sort(points.begin(), points.end(), [](Point a, Point b) {
        return a.x < b.x;
        });
    if (points[0].y < points[1].y) {
        newPoints[0] = points[0]; 
        newPoints[3] = points[1]; 
    }
    else {
        newPoints[0] = points[1]; 
        newPoints[3] = points[0];
    }

    if (points[2].y < points[3].y) {
        newPoints[1] = points[2]; 
        newPoints[2] = points[3]; 
    }
    else {
        newPoints[1] = points[3]; 
        newPoints[2] = points[2]; 
    }
    return newPoints;
}


void main() {
    string path = "Resources/paper.jpg"; 
    Mat img = imread(path);
    resize(img, img, Size(), 0.5, 0.5);
    Mat imgGray, imgBlur, mask, imgWarp;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(5, 5), 1, 1);
    threshold(imgBlur, mask, 180, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int max_idx = -1;
    double max_area = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 10000 && area > max_area) { 
            max_area = area;
            max_idx = i;
        }
    }
    vector<Point> approx_points;
    float peri = arcLength(contours[max_idx], true);
    approxPolyDP(contours[max_idx], approx_points, 0.02 * peri, true);
    if (approx_points.size() == 4) {
        vector<Point2f> sorted_corners = reorder(approx_points);
        Point2f src[4];
        for (int i = 0; i < 4; ++i) {
            src[i] = sorted_corners[i];
        }
        Mat matrix;
        int h = 700;
        int w = (int)(h * (210.0f / 297.0f));
        Point2f dst[4] = { {0.0f, 0.0f}, {(float)w, 0.0f}, {(float)w, (float)h}, {0.0f, (float)h} };
        matrix = getPerspectiveTransform(src, dst);
        warpPerspective(img, imgWarp, matrix, Point(w, h));
        imshow("Doc Perspective", imgWarp);
        for (int i = 0; i < 4; ++i) {
            circle(img, Point(src[i].x, src[i].y), 5, Scalar(0, 0, 255), FILLED);
        }
    }
    imshow("Image", img);
    waitKey(0);
}
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace cv;
using namespace std;
double RAD_TO_DEG = 180.0 / CV_PI;
double ANGLE_TOLERANCE = 10.0;
double MIN_CONTOUR_AREA = 300.0;
double AREA_RATIO_DUPLICATE_THRESHOLD = 0.6; 
double CENTROID_DISTANCE_THRESHOLD = 5.0; 
double VERTEX_PROXIMITY_MERGE_THRESHOLD = 5.0;
double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    double dot_product = (dx1 * dx2 + dy1 * dy2);
    double cross_norm = sqrt(dx1 * dx1 + dy1 * dy1) * sqrt(dx2 * dx2 + dy2 * dy2);
    if (cross_norm == 0) return 0.0;
    double cos_theta = dot_product / cross_norm;
    cos_theta = min(max(cos_theta, -1.0), 1.0);
    return acos(cos_theta) * RAD_TO_DEG;
}

Point calculateCentroid(const vector<Point>& contour) {
    Moments M = moments(contour);
    if (M.m00 == 0) return Point(-1, -1);
    return Point((M.m10 / M.m00), (M.m01 / M.m00));
}

double distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void displayTriangleCount(Mat& img, int obtuse, int acute, int right) {
    int line_height = 25;
    int x_start = 10;
    int y_start = 30;

    stringstream ss_obtuse, ss_acute, ss_right;
    ss_obtuse << "Obtuse: " << obtuse;
    ss_acute << "Acute: " << acute;
    ss_right << "Right: " << right;

    Rect rect(x_start - 5, y_start - line_height - 5, 200, 3 * line_height + 5);
    Mat roi = img(rect);
    addWeighted(roi, 0.5, roi, 0.5, 0, roi);

    putText(img, ss_obtuse.str(), Point(x_start, y_start),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

    putText(img, ss_acute.str(), Point(x_start, y_start + line_height),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

    putText(img, ss_right.str(), Point(x_start, y_start + 2 * line_height),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
}
bool canBeTreatedAsTriangle(vector<Point>& approx) {
    if (approx.size() == 3) {
        return true;
    }
    return false;
}

void getContours(Mat imgDil, Mat& img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int count_obtuse = 0;
    int count_acute = 0;
    int count_right = 0;

    findContours(imgDil, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat imgContour = img.clone();
    vector<double> areas(contours.size());
    vector<Point> centroids(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        areas[i] = contourArea(contours[i]);
        centroids[i] = calculateCentroid(contours[i]);
    }

    for (size_t i = 0; i < contours.size(); i++) {
        double area = areas[i];
        if (area < MIN_CONTOUR_AREA) continue;

        bool isValidContour = true;
        int currentParent = hierarchy[i][3]; 
        if (currentParent != -1) {
            double parentArea = areas[currentParent];
            Point parentCentroid = centroids[currentParent];
            Point currentCentroid = centroids[i];

            double areaRatio = area / parentArea;
            double centroidDist = distance(currentCentroid, parentCentroid);
            if (areaRatio >= AREA_RATIO_DUPLICATE_THRESHOLD) {
                isValidContour = false;
            }
            else if (centroidDist < CENTROID_DISTANCE_THRESHOLD) {
                isValidContour = false;
            }
        }

        if (!isValidContour) continue;

        double perimeter = arcLength(contours[i], true);
        double epsilon = 0.04 * perimeter;
        vector<Point> approx;
        approxPolyDP(contours[i], approx, epsilon, true);
        drawContours(imgContour, contours, (int)i, Scalar(255, 0, 255), 2);
        if (canBeTreatedAsTriangle(approx)) {
            double angle0 = angle(approx[1], approx[2], approx[0]);
            double angle1 = angle(approx[0], approx[2], approx[1]);
            double angle2 = angle(approx[0], approx[1], approx[2]);

            double maxAngle = max({ angle0, angle1, angle2 });

            string shape_label = "Triangle";
            string type_label = "";
            Scalar text_color;

            if (abs(maxAngle - 90.0) < ANGLE_TOLERANCE) {
                type_label = " (Right)";
                text_color = Scalar(0, 255, 0);
                count_right++;
            }
            else if (maxAngle > 90.0) {
                type_label = " (Obtuse)";
                text_color = Scalar(0, 0, 255);
                count_obtuse++;
            }
            else {
                type_label = " (Acute)";
                text_color = Scalar(255, 0, 0);
                count_acute++;
            }

            int nestingLevel = 0;
            int tempParent = hierarchy[i][3];
            while (tempParent != -1) {
                nestingLevel++;
                tempParent = hierarchy[tempParent][3];
            }

            Scalar draw_color;
            if (nestingLevel == 0) {
                draw_color = Scalar(255, 0, 255);
            }
            else if (nestingLevel == 1) {
                draw_color = Scalar(0, 255, 255); 
            }
            else if (nestingLevel == 2) {
                draw_color = Scalar(255, 255, 0); 
            }
            else {
                draw_color = Scalar(0, 165, 255);
            }

            drawContours(img, contours, (int)i, draw_color, 3);

            Point center = centroids[i];
            putText(img, shape_label + type_label, center, FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        }
    }

    displayTriangleCount(img, count_obtuse, count_acute, count_right);
    imshow("All Contours (Canny + Dilate)", imgContour);
}

void processImageWithCanny(string& path) {
    Mat img = imread(path);

    Mat imgResized;
    resize(img, imgResized, Size(), 0.75, 0.75, INTER_LINEAR);

    Mat imgGray, imgBlur, imgCanny, imgDil;
    cvtColor(imgResized, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(5, 5), 3, 0);

    Canny(imgBlur, imgCanny, 70, 150);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgCanny, imgDil, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    imshow("2. MORPH_CLOSE (Aggressive)", imgDil);

    getContours(imgDil, imgResized);

    imshow("Original (with Triangles, Resized)", imgResized);
    waitKey(0);
}

int main() {
    string path_example = "Resources/triangle_*.jpg";
    for (int i = 0; i < 10; i++) {
        path_example[19] = i + '0';
        processImageWithCanny(path_example);
    }
    return 0;
}
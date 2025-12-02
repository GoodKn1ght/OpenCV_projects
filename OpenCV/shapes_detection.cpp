#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
void drawBounding(Mat img, Rect& bbox, string& label) {
    rectangle(img, bbox.tl(), bbox.br(), Scalar(0, 255, 0), 2); 
    Point textPos(bbox.x, bbox.y - 5);
    if (bbox.y < 15) { 
        textPos = Point(bbox.x + 5, bbox.y + 15);
    }
    putText(img, label, textPos, FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
}
void getContours(Mat imgDil, Mat img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);

        if (area > 1000) { 
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.01 * peri, true);
            boundRect[i] = boundingRect(conPoly[i]);
            int cornerCount = conPoly[i].size();
            string objectType;
            if (cornerCount == 3) {
                objectType = "Triangle";
            }
            else if (cornerCount == 4) {
                float aspectRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                if (aspectRatio > 0.95 && aspectRatio < 1.05) {
                    objectType = "Square";
                }
                else {
                    objectType = "Rectangle";
                }
            }
            else if (cornerCount > 4) {
                objectType = "Circle"; 
            }
            else {
                objectType = "Unknown";
            }
            drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
            drawBounding(img, boundRect[i], objectType);
        }
    }
}

void main() {
    string path = "Resources/shapes.png";
    Mat img = imread(path);

    Mat imgGray, imgBlur, imgCanny, imgDil;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(5, 5), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);
    Mat imgCopy = img.clone();
    getContours(imgDil, imgCopy);
    imshow("Original Image with Detections", imgCopy);
    imshow("Canny Edges", imgCanny);
    imshow("Dilation", imgDil);
    waitKey(0);
}
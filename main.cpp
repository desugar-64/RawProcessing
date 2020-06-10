#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/Align.h"

using namespace cv;

void processRaw() {
    Demosaic demosaic("/home/sergeyfitis/raw/flower.dng", "/home/sergeyfitis/raw/flower_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
//    processRaw();
    std::cout << "Ok!" << std::endl;
    Mat rect1 = imread("rect1.png", IMREAD_GRAYSCALE);
    Mat rect2 = imread("rect2.png", IMREAD_GRAYSCALE);

    double a = compare(rect1, rect2);
    
    cv::namedWindow("rect1", CV_WINDOW_FREERATIO);
    cv::namedWindow("rect2", CV_WINDOW_FREERATIO);
    cv::imshow("rect1", rect1);
    cv::imshow("rect2", rect2);

    cv::waitKey(0);
    return 0;
}


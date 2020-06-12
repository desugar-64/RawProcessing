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
    Mat base = imread("../align_test/flower_base.png", IMREAD_GRAYSCALE);
    Mat shifted = imread("../align_test/flower_shifted.png", IMREAD_GRAYSCALE);
    int width = base.cols;
    int height = base.rows;

    int searchSize = 120;
    const Point2i shiftOffset = comparePyramidLayer(
            base,
            shifted,
            0,
            Range(-1 * searchSize, 1 * searchSize),
            Range(-1 * searchSize, 1 * searchSize)
    );

    Mat correction = translateImg(shifted, shiftOffset.x, shiftOffset.y);
    Mat aligned;
    cv::addWeighted(base, 0.5, correction, 0.5, 0., aligned);

//    compare(base, shifted, -100, -60);


    cv::namedWindow("base", CV_WINDOW_FREERATIO);
    cv::namedWindow("shifted", CV_WINDOW_FREERATIO);
    cv::namedWindow("aligned", CV_WINDOW_FREERATIO);
    cv::imshow("base", base);
    cv::imshow("shifted", shifted);
    cv::imshow("aligned", aligned);

    cv::waitKey(0);
    return 0;
}


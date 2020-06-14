#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/Align.h"

using namespace cv;

void processRaw() {
    Demosaic demosaic("/home/sergeyfitis/raw/flower.dng", "/home/sergeyfitis/raw/wall_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
//    processRaw();
    std::cout << "Ok!" << std::endl;
    Mat base = imread("../align_test/wall_base.png", IMREAD_GRAYSCALE);
    Mat baseColor = imread("../align_test/wall_base.png", IMREAD_COLOR);
    Mat shifted = imread("../align_test/wall_shifted.png", IMREAD_GRAYSCALE);
    Mat shiftedColor = imread("../align_test/wall_shifted.png", IMREAD_COLOR);

    const Point2d &alignment = align(base, shifted);
    auto xAlignment = alignment.x;
    auto yAlignment = alignment.y;

    Mat corrected = translateImg(shiftedColor, xAlignment, yAlignment);
    namedWindow("corrected", WINDOW_NORMAL);
    namedWindow("aligned", WINDOW_NORMAL);
    Mat aligned;
    addWeighted(baseColor, 0.5, corrected, 0.5, 0, aligned);
    imshow("aligned", aligned);

    Mat noAlign;
    addWeighted(baseColor, 0.5, shiftedColor, 0.5, 0, noAlign);
    imshow("corrected", corrected);
    imwrite("no_align.png", noAlign);
    imwrite("align.png", aligned);

    cv::waitKey(0);
    return 0;
}


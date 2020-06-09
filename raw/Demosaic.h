//
// Created by sergeyfitis on 03.06.20.
//

#ifndef TESTLIBRAW_DEMOSAIC_H
#define TESTLIBRAW_DEMOSAIC_H

#include "string"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "libraw/libraw.h"
#include "CfaPattern.h"
#include "filter/Filter.h"

class Demosaic {
private:
    LibRaw *libRaw;
    Filter *bayerFilter;
    cv::Mat demosaicImage;
    cv::Mat colorImage;
    cv::Mat result;

    static std::string getCfaPatternString(LibRaw *processor);
    static CfaPattern getCfaPattern(LibRaw *processor);
    static cv::Mat getColorCorrectionMatrix(LibRaw *processor);

public:
    explicit Demosaic(const char *filePath, const char *originalImgPath);

    void generateRGBComponents();

    void interpolateGRBG();

    void colorize();

    void squaredDifference();

    void display();
};

#endif //TESTLIBRAW_DEMOSAIC_H

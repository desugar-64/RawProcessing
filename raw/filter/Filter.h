//
// Created by sergeyfitis on 09.06.20.
//

#ifndef RAWPROCESSING_FILTER_H
#define RAWPROCESSING_FILTER_H

#include "opencv2/opencv.hpp"
#include "../CfaPattern.h"

class Filter {

protected:
    virtual ~Filter() = default;

public:
    cv::Mat rawImage;
    cv::Mat bayerGreen;
    cv::Mat bayerRed;
    cv::Mat bayerBlue;

    int rows;
    int cols;

    virtual void generateComponents() = 0;

    virtual void interpolate() = 0;
};

#endif //RAWPROCESSING_FILTER_H

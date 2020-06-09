//
// Created by sergeyfitis on 10.06.20.
//

#ifndef RAWPROCESSING_FILTERGRBG_H
#define RAWPROCESSING_FILTERGRBG_H

#include "Filter.h"

class FilterGRBG: public Filter {
public:
    explicit FilterGRBG(ushort *rawPixels, int width, int height);
    void generateComponents() override;
    void interpolate() override;

    ~FilterGRBG() override;

    static void write(const char *name, const cv::Mat &img);
};

#endif //RAWPROCESSING_FILTERGRBG_H

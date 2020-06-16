//
// Created by sergeyfitis on 15.06.20.
//

#ifndef RAWPROCESSING_TILE_H
#define RAWPROCESSING_TILE_H

#include "opencv2/opencv.hpp"

class Tile {
public:
    cv::Rect2i rect;
    cv::Rect2i alignmentOffset;
    double alignmentError = 0.;

    explicit Tile(cv::Rect2i &rect);
};

#endif //RAWPROCESSING_TILE_H

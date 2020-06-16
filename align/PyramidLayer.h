//
// Created by sergeyfitis on 16.06.20.
//

#ifndef RAWPROCESSING_PYRAMIDLAYER_H
#define RAWPROCESSING_PYRAMIDLAYER_H

#include "opencv2/opencv.hpp"
#include "Tile.h"

class PyramidLayer {
public:
    cv::Mat layer;
    int number;
    std::vector<Tile> tiles;
    explicit PyramidLayer(cv::Mat &layer, int number);
    void generateTiles();
};


#endif //RAWPROCESSING_PYRAMIDLAYER_H

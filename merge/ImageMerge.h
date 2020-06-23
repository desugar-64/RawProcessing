//
// Created by sergeyfitis on 24.06.20.
//

#ifndef RAWPROCESSING_IMAGEMERGE_H
#define RAWPROCESSING_IMAGEMERGE_H

#include "opencv2/opencv.hpp"
#include "../align/Tile.h"


class ImageMerge {
public:
    explicit ImageMerge(cv::Mat &baseImage);
    cv::Mat merge(
            cv::Mat &base,
            cv::Mat &align,
            std::vector<Tile> &alignmentOffsets,
            uchar numberOfImages);

private:
    cv::Mat &baseImage;
};


#endif //RAWPROCESSING_IMAGEMERGE_H

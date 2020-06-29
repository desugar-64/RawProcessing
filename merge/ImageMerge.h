//
// Created by sergeyfitis on 24.06.20.
//

#ifndef RAWPROCESSING_IMAGEMERGE_H
#define RAWPROCESSING_IMAGEMERGE_H

#define ALIGNMENT_PRECISION 2.4
#define DEBUG_TILES true

#include "opencv2/opencv.hpp"
#include "../align/Tile.h"


class ImageMerge {
public:
    void  merge(
            cv::Mat &merged,
            cv::Mat &base,
            cv::Mat &align,
            std::vector<Tile> &alignmentOffsets,
            uchar numberOfImages);

    cv::Mat mergeBurst(cv::Mat &base, std::vector<cv::Mat> burst, std::map<int, std::vector<Tile>> &alignmentOffsets);

private:
    static void mergeAveraged(
            cv::Mat &mergeTo, cv::Mat &base, cv::Mat &align, uchar numberOfImages);
};


#endif //RAWPROCESSING_IMAGEMERGE_H

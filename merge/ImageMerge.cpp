//
// Created by sergeyfitis on 24.06.20.
//

#include "ImageMerge.h"

using namespace cv;

Mat ImageMerge::merge(Mat &base, Mat &align, std::vector<Tile> &alignmentOffsets, uchar numberOfImages) {

}

ImageMerge::ImageMerge(Mat &baseImage) : baseImage(baseImage) {}

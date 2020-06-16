//
// Created by sergeyfitis on 15.06.20.
//

#ifndef RAWPROCESSING_IMAGEALIGN_H
#define RAWPROCESSING_IMAGEALIGN_H

#define PYRAMID_LAYERS 4
#define COMPARE_EVERY_N_PIXEL 10
#define BASE_SEARCH_DISTANCE 10
#define T_SIZE 32           // Size of a tile in the bayer mosaiced image
#define T_SIZE_2 16         // Half of T_SIZE and the size of a tile throughout the alignment pyramid

#define MIN_OFFSET -168     // Min total alignment (based on three levels and downsampling by 4)
#define MAX_OFFSET 126      // Max total alignment. Differs from MIN_OFFSET because total search range is 8 for better vectorization

#define DOWNSAMPLE_RATE_PER_LEVEL 4   // Rate at which layers of the alignment pyramid are downsampled relative to each other

#include "opencv2/opencv.hpp"
#include "Tile.h"
#include "PyramidLayer.h"

class ImageAlign {
private:
    std::vector<PyramidLayer> gaussianPyramid(cv::Mat &base);

    cv::Point2d shiftByPhaseCorrelation(const cv::Mat &source, const cv::Mat &target, bool bUseHanningWindow);

    double compare(cv::Mat &base, cv::Mat &other, int xOffset, int yOffset);

    cv::Point2d
    comparePyramidLayerFFT(cv::Mat &base, cv::Mat &layer, int layerNumber, double prevAlignmentX,
                           double prevAlignmentY);

    cv::Point2i
    comparePyramidLayer(PyramidLayer &base, PyramidLayer &layer, cv::Range xSearchRange, cv::Range ySearchRange,
                        int prevAlignmentX,
                        int prevAlignmentY);

    std::vector<Tile> generateTilesForLayer(int layerNumber, cv::Mat &layer);

public:
    cv::Mat translateImg(cv::Mat &img, double xOffset, double yOffset);
    cv::Point2d align(cv::Mat &base, cv::Mat &shifted);
};


#endif //RAWPROCESSING_IMAGEALIGN_H

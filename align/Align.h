//
// Created by sergeyfitis on 11.06.20.
//

#define T_SIZE 32           // Size of a tile in the bayer mosaiced image
#define T_SIZE_2 16         // Half of T_SIZE and the size of a tile throughout the alignment pyramid

#define MIN_OFFSET -168     // Min total alignment (based on three levels and downsampling by 4)
#define MAX_OFFSET 126      // Max total alignment. Differs from MIN_OFFSET because total search range is 8 for better vectorization

#define DOWNSAMPLE_RATE 4   // Rate at which layers of the alignment pyramid are downsampled relative to each other

#include "opencv2/opencv.hpp"

using namespace cv;

/*
 * returns size of difference between two grayscale frames
 * */
static double compare(Mat& base, Mat& other) {
    assert(base.channels() == 1); // image must be grayscale
    assert(other.channels() == 1);

    Mat diff;
    absdiff(base, other, diff);
    auto mean = cv::mean(diff);
    return sum(mean).val[0];
}

static Point2d comparePyramidLayer(Mat& base, Mat& layer, int layerNumber) {
    // TODO
}

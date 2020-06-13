//
// Created by sergeyfitis on 11.06.20.
//

#define PYRAMID_LAYERS 3
#define T_SIZE 32           // Size of a tile in the bayer mosaiced image
#define T_SIZE_2 16         // Half of T_SIZE and the size of a tile throughout the alignment pyramid

#define MIN_OFFSET -168     // Min total alignment (based on three levels and downsampling by 4)
#define MAX_OFFSET 126      // Max total alignment. Differs from MIN_OFFSET because total search range is 8 for better vectorization

#define DOWNSAMPLE_RATE_PER_LEVEL 4   // Rate at which layers of the alignment pyramid are downsampled relative to each other

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

vector<Mat> gaussianPyramid(Mat &base) {
    vector<Mat> pyramid;
    pyramid.push_back(base);
    Mat pyrLayer = base;
    for (int layer = 0; layer < PYRAMID_LAYERS; layer++) {
        pyrDown(pyrLayer, pyrLayer);
        pyramid.push_back(pyrLayer);
    }
//    for (int i = 0; i < pyramid.size(); i++) {
//        Mat mat = pyramid[i];
//        const String window = format("layer%d, (%d)x(%d)", i, mat.cols, mat.rows);
//        namedWindow(window, WINDOW_NORMAL);
//        imshow(window, mat);
//    }
    return pyramid;
}

Mat translateImg(Mat &img, int xOffset, int yOffset) {
    Mat out;
    Mat transMat = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
    warpAffine(img, out, transMat, img.size());

//    const String winname = format("out=%dx%d", xOffset, yOffset);
//    namedWindow(winname, WINDOW_NORMAL);
//    imshow(winname, out);
    return out;
}

/*
 * returns size of difference between two grayscale frames
 * */
static double compare(Mat &base, Mat &other, int xOffset, int yOffset) {
    assert(base.channels() == 1); // image must be grayscale
    assert(other.channels() == 1);

    auto compare = translateImg(other, xOffset, yOffset);
//    namedWindow("compare", WINDOW_NORMAL);
//    imshow("compare", compare);

    Mat diff;
    absdiff(base, compare, diff);
    diff = abs(base - compare);
//    namedWindow("diff", WINDOW_NORMAL);
//    imshow("diff", diff);
    auto meanValue = mean(diff);
//    waitKey(0);
    return sum(meanValue).val[0];
}

static Point2i comparePyramidLayer(Mat &base, Mat &layer, int layerNumber, Range xSearchRange, Range ySearchRange, int prevAlignmentX, int prevAlignmentY) {

    std::printf("comparePyramidLayer=%d, (%d)x(%d)\n", layerNumber, base.cols, base.rows);
    Point2i *alignShift = nullptr;
    double minAlignmentError = std::numeric_limits<double>::max();
    for (int x = xSearchRange.start; x <= xSearchRange.end; x++) {
        for (int y = ySearchRange.start; y < ySearchRange.end; y++) {
            if (alignShift == nullptr) {
                auto align = Point2i(x, y);
                alignShift = &align;
            }
            double alignmentError = compare(base, layer, x + prevAlignmentX, y + prevAlignmentY);
            if (alignmentError < minAlignmentError) {
                minAlignmentError = alignmentError;
                alignShift->x = x + prevAlignmentX;
                alignShift->y = y + prevAlignmentY;
            }
        }
    }
    std::printf("minAlignmentError=%f\n", minAlignmentError);
    std::printf("alignOffset=(%d)x(%d)\n", alignShift->x, alignShift->y);
    std::printf("--------------------------\n");
    return *alignShift;
}


Point2i align(Mat &base, Mat &shifted) {
    auto basePyr = gaussianPyramid(base);
    auto shiftedPyr = gaussianPyramid(shifted);
    assert(basePyr.size() == shiftedPyr.size());

    auto minSearchDistance = 24; // min search range 24 pixels
    auto xAlignment = 0;
    auto yAlignment = 0;
    auto xSearchRange = Range(-1 * minSearchDistance, 1 * minSearchDistance);
    auto ySearchRange = Range(-1 * minSearchDistance, 1 * minSearchDistance);

    auto initialAlignment = comparePyramidLayer(basePyr[basePyr.size() - 1],
                                                shiftedPyr[shiftedPyr.size() - 1],
                                                basePyr.size() - 1,
                                                xSearchRange,
                                                ySearchRange,
                                                xAlignment,
                                                yAlignment);
    xAlignment = initialAlignment.x;
    yAlignment = initialAlignment.y;


    for (int layer = basePyr.size() - 2; layer > 0; layer--) { // we do not aligning 0 level layers.. so skipping them.
        auto baseLayer = basePyr[layer];
        auto shiftedLayer = shiftedPyr[layer];
        xAlignment *= 2;
        yAlignment *= 2;
        auto scale = 1;
        xSearchRange.start = -1 * (minSearchDistance * scale);
        xSearchRange.end = 1 * (minSearchDistance * scale);
        ySearchRange.start = -1 * (minSearchDistance * scale);
        ySearchRange.end = 1 * (minSearchDistance * scale);
        Point2i alignOffset = comparePyramidLayer(baseLayer, shiftedLayer, layer, xSearchRange, ySearchRange, xAlignment, yAlignment);
        xAlignment = alignOffset.x;
        yAlignment = alignOffset.y;
    }

    xAlignment *= 2;
    yAlignment *= 2;

    std::printf("finalAlignment=%dx%d\n", xAlignment, yAlignment);

    return {xAlignment, yAlignment};
}
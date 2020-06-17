//
// Created by sergeyfitis on 15.06.20.
//

#include <opencv2/imgproc/types_c.h>
#include "ImageAlign.h"

using namespace cv;
using namespace std;

std::vector<PyramidLayer> ImageAlign::gaussianPyramid(cv::Mat &base) {
    vector<PyramidLayer> pyramid;
    PyramidLayer layer0 = PyramidLayer(base, 0);
    layer0.generateTiles();
    pyramid.push_back(layer0);
    Mat pyrLayer = base;
    for (int layer = 1; layer < PYRAMID_LAYERS; layer++) {
        pyrDown(pyrLayer, pyrLayer);
        PyramidLayer layerN = PyramidLayer(pyrLayer, layer);
        layerN.generateTiles();
        pyramid.push_back(layerN);
    }
//    for (int i = 0; i < pyramid.size(); i++) {
//        Mat mat = pyramid[i];
//        const String window = format("layer%d, (%d)x(%d)", i, mat.cols, mat.rows);
//        namedWindow(window, WINDOW_NORMAL);
//        imshow(window, mat);
//    }
    return pyramid;
}

cv::Mat ImageAlign::translateImg(Mat &img, int xOffset, int yOffset) {
    Mat out;
    Mat transMat = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
    warpAffine(img, out, transMat, img.size(), CV_INTER_LINEAR, BORDER_CONSTANT, Scalar());
// You can try more different parameters
//    warpAffine(img, out, M, dsize, CV_INTER_LINEAR, BORDER_TRANSPARENT, Scalar());


//    const String winname = format("out=%dx%d", xOffset, yOffset);
//    namedWindow(winname, WINDOW_NORMAL);
//    imshow(winname, out);
//    waitKey(0);
    return out;
}

cv::Point2d ImageAlign::shiftByPhaseCorrelation(const Mat &base, const Mat &layer, bool bUseHanningWindow) {
    CV_Assert(layer.type() == CV_32FC1);
    CV_Assert(base.type() == CV_32FC1);

    Point2d shift;
    if (bUseHanningWindow) {
        Mat hann;
        createHanningWindow(hann, layer.size(), CV_32F);
        shift = phaseCorrelate(layer, base, hann);
    } else {
        shift = phaseCorrelate(layer, base);
    }

//    cout << "Detected shift: " << shift << endl;

    return shift;
}

double ImageAlign::compare(Mat &base, Mat &other, double xOffset, double yOffset) {
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

cv::Point2d ImageAlign::comparePyramidLayerFFT(Mat &base,
                                               Mat &layer,
                                               double prevAlignmentX,
                                               double prevAlignmentY) {
    Point2d alignmentShift;

    Mat base32F;
    Mat layer32F;
    Mat shiftLayer = translateImg(layer, prevAlignmentX, prevAlignmentY);
    base.convertTo(base32F, CV_32FC1);
    shiftLayer.convertTo(layer32F, CV_32FC1);
    const Point2d point = shiftByPhaseCorrelation(base32F, layer32F, true);
    alignmentShift.x = point.x;
    alignmentShift.y = point.y;
//    cout << "Detected shift: " << alignmentShift << endl;
//    std::printf("--------------------------\n");
    return alignmentShift;
}

void ImageAlign::comparePyramidLayer(PyramidLayer &previousShiftedLayer,
                                     PyramidLayer &base,
                                     PyramidLayer &shifted,
                                     bool useFFT) {

    CV_Assert(base.tiles.size() == shifted.tiles.size());
    CV_Assert(base.tiles[0].rect.width == shifted.tiles[0].rect.width);
    CV_Assert(base.tiles[0].rect.height == shifted.tiles[0].rect.height);

    if (shifted.number < PYRAMID_LAYERS - 1) {
        CV_Assert(shifted.number < previousShiftedLayer.number);
    }


    const double maxDouble = std::numeric_limits<double>::max();

    std::printf("comparePyramidLayer=%d, %dx%d, useFFT=%d\n", base.number, base.layer.cols, base.layer.rows, useFFT);
    Point2d alignmentShift(maxDouble, maxDouble);
    int doneIterations = 0;
    int multipleOf = COMPARE_EVERY_N_PIXEL;

    Range searchRange(-ALIGN_SEARCH_DISTANCE, ALIGN_SEARCH_DISTANCE);

    for (int tileIndex = 0; tileIndex < base.tiles.size(); tileIndex++) {
        auto &previousShiftedLayerTile = previousShiftedLayer.tiles[tileIndex];
        auto &baseLayerTile = base.tiles[tileIndex];
        auto &shiftedLayerTile = shifted.tiles[tileIndex];

        double minAlignmentError = maxDouble;


        double prevShiftedLayerTileAlignX = previousShiftedLayerTile.alignmentOffset.x * 2;
        double prevShiftedLayerTileAlignY = previousShiftedLayerTile.alignmentOffset.y * 2;

        auto baseTileMat = base.layer(baseLayerTile.rect);
        auto shiftedTileMat = shifted.layer(shiftedLayerTile.rect);

        if (useFFT) {
            auto fftShift = comparePyramidLayerFFT(baseTileMat,
                                                   shiftedTileMat,
                                                   prevShiftedLayerTileAlignX,
                                                   prevShiftedLayerTileAlignY);
            alignmentShift = fftShift;
            minAlignmentError = compare(baseTileMat, shiftedTileMat, fftShift.x, fftShift.y);
        } else {
            auto tileSearchRange = min(baseLayerTile.rect.width, ALIGN_SEARCH_DISTANCE);
            searchRange.start = -tileSearchRange;
            searchRange.end = tileSearchRange;
            for (int x = searchRange.start; x <= searchRange.end; x++) {
                for (int y = searchRange.start; y < searchRange.end; y++) {
                    if (searchRange.size() > multipleOf && x % multipleOf != 0 && y % multipleOf != 0) {
                        continue;
                    }
                    doneIterations++;

                    double alignmentError = compare(baseTileMat,
                                                    shiftedTileMat,
                                                    x + prevShiftedLayerTileAlignX,
                                                    y + prevShiftedLayerTileAlignY);
                    if (alignmentError < minAlignmentError) {
                        minAlignmentError = alignmentError;
                        alignmentShift.x = x + prevShiftedLayerTileAlignX;
                        alignmentShift.y = y + prevShiftedLayerTileAlignY;
                    }
                }
            }

        }

        shiftedLayerTile.alignmentOffset.x = alignmentShift.x;
        shiftedLayerTile.alignmentOffset.y = alignmentShift.y;
        shiftedLayerTile.alignmentError = minAlignmentError;

    }
    std::printf("processed %d iterations on layer=%d\n", doneIterations, base.number);
    std::printf("--------------------------\n");
}

std::vector<Tile> ImageAlign::align(Mat &base, Mat &shifted) {
    auto start = std::chrono::high_resolution_clock::now();

    auto basePyr = gaussianPyramid(base);
    auto shiftedPyr = gaussianPyramid(shifted);
    assert(basePyr.size() == shiftedPyr.size());

    auto &basePyramidTop = basePyr[basePyr.size() - 1];
    auto &shiftedPyramidTop = shiftedPyr[shiftedPyr.size() - 1];

    comparePyramidLayer(shiftedPyramidTop, basePyramidTop, shiftedPyramidTop);

    for (int layer = basePyr.size() - 2; layer > 0; layer--) { // we do not aligning 0 level layers.. so skipping them.
        auto &baseLayer = basePyr[layer];
        auto &shiftedLayer = shiftedPyr[layer];
        auto &prevShiftedLayer = shiftedPyr[layer + 1];
        comparePyramidLayer(prevShiftedLayer, baseLayer, shiftedLayer, layer < 3);
    }

    auto &shiftedPyramidLayer1 = shiftedPyr[1];

    // interpolate tile align offsets to the final 0 layer
    for (auto &tile : shiftedPyramidLayer1.tiles) {
        tile.alignmentOffset.x *= 2;
        tile.alignmentOffset.y *= 2;
        tile.rect.x *= 2;
        tile.rect.y *= 2;
        tile.rect.width *= 2;
        tile.rect.height *= 2;
    }

    /*auto finalShift = comparePyramidLayer(basePyr[0], shiftedPyr[0], 0, xSearchRange, ySearchRange, xAlignment,
                                          yAlignment);
    xAlignment = finalShift.x;
    yAlignment = finalShift.y;*/

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::printf("duration=%ld.ms\n", duration.count() / 1000);
    return shiftedPyramidLayer1.tiles;
}

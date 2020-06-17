//
// Created by sergeyfitis on 15.06.20.
//

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

cv::Mat ImageAlign::translateImg(Mat &img, double xOffset, double yOffset) {
    Mat out;
    Mat transMat = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
    warpAffine(img, out, transMat, img.size());

//    const String winname = format("out=%dx%d", xOffset, yOffset);
//    namedWindow(winname, WINDOW_NORMAL);
//    imshow(winname, out);
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

    cout << "Detected shift: " << shift << endl;

    return shift;
}

double ImageAlign::compare(Mat &base, Mat &other, int xOffset, int yOffset) {
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

cv::Point2d ImageAlign::comparePyramidLayerFFT(Mat &base, Mat &layer, int layerNumber, double prevAlignmentX,
                                               double prevAlignmentY) {
    std::printf("comparePyramidLayerFFT=%d, %dx%d\n", layerNumber, base.cols, base.rows);
    Point2d alignmentShift;

    Mat base32F;
    Mat layer32F;
    Mat shiftLayer = translateImg(layer, prevAlignmentX, prevAlignmentY);
    base.convertTo(base32F, CV_32FC1);
    shiftLayer.convertTo(layer32F, CV_32FC1);
    const Point2d point = shiftByPhaseCorrelation(base32F, layer32F, true);
    alignmentShift.x = point.x;
    alignmentShift.y = point.y;
    cout << "Detected shift: " << alignmentShift << endl;
    std::printf("--------------------------\n");
    return alignmentShift;
}

void ImageAlign::comparePyramidLayer(PyramidLayer &previousShiftedLayer,
                                     PyramidLayer &base,
                                     PyramidLayer &shifted) {

    CV_Assert(base.tiles.size() == shifted.tiles.size());
    CV_Assert(base.tiles[0].rect.width == shifted.tiles[0].rect.width);
    CV_Assert(base.tiles[0].rect.height == shifted.tiles[0].rect.height);

    std::printf("comparePyramidLayer=%d, %dx%d\n", base.number, base.layer.cols, base.layer.rows);
    Point2i alignmentShift(0, 0);
    int doneIterations = 0;
    int multipleOf = COMPARE_EVERY_N_PIXEL;

    Range searchRange(-ALIGN_SEARCH_DISTANCE, ALIGN_SEARCH_DISTANCE);

    for (int tileIndex = 0; tileIndex < base.tiles.size(); tileIndex++) {
        auto &previousShiftedLayerTile = previousShiftedLayer.tiles[tileIndex];
        auto &baseLayerTile = base.tiles[tileIndex];
        auto &shiftedLayerTile = shifted.tiles[tileIndex];

        double minAlignmentError = std::numeric_limits<double>::max();

        auto tileSearchRange = min(baseLayerTile.rect.width, ALIGN_SEARCH_DISTANCE);
        searchRange.start = -tileSearchRange;
        searchRange.end = tileSearchRange;

        int prevShiftedLayerTileAlignX = previousShiftedLayerTile.alignmentOffset.x * 2;
        int prevShiftedLayerTileAlignY = previousShiftedLayerTile.alignmentOffset.y * 2;

        auto baseTileMat = base.layer(baseLayerTile.rect);
        auto shiftedTileMat = shifted.layer(shiftedLayerTile.rect);
//        visualizeMat(
//                format("base tile %d, x: %d, y: %d, w: %d, h: %d", tileIndex, baseLayerTile.rect.x, baseLayerTile.rect.y,
//                       baseLayerTile.rect.width, baseLayerTile.rect.height), baseTileMat);
//        visualizeMat(
//                format("shifted tile %d, x: %d, y: %d, w: %d, h: %d", tileIndex, shiftedLayerTile.rect.x,
//                       shiftedLayerTile.rect.y,
//                       shiftedLayerTile.rect.width, shiftedLayerTile.rect.height), shiftedTileMat, true);

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

    /* auto initialAlignment = comparePyramidLayerFFT(basePyr[basePyr.size() - 1],
                                                    shiftedPyr[shiftedPyr.size() - 1],
                                                    basePyr.size() - 1,
                                                    xAlignment,
                                                    yAlignment);*/

    for (int layer = basePyr.size() - 2; layer > 0; layer--) { // we do not aligning 0 level layers.. so skipping them.
        auto &baseLayer = basePyr[layer];
        auto &shiftedLayer = shiftedPyr[layer];
        auto &prevShiftedLayer = shiftedPyr[layer + 1];
        comparePyramidLayer(prevShiftedLayer, baseLayer, shiftedLayer);

        /*Point2d alignOffset = comparePyramidLayerFFT(baseLayer,
                                                     shiftedLayer,
                                                     layer,
                                                     xAlignment,
                                                     yAlignment);*/

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

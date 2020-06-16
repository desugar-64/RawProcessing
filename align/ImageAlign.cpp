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

cv::Point2i
ImageAlign::comparePyramidLayer(PyramidLayer &base, PyramidLayer &layer, cv::Range xSearchRange, cv::Range ySearchRange,
                                int prevAlignmentX, int prevAlignmentY) {

    std::printf("comparePyramidLayer=%d, %dx%d\n", base.number, base.layer.cols, base.layer.rows);
    Point2i *alignmentShift = nullptr;
    double minAlignmentError = std::numeric_limits<double>::max();
    int doneIterations = 0;
    int multipleOf = COMPARE_EVERY_N_PIXEL;

    auto iterateTiles = [](Tile &tile) {
        // TODO
    };

    for_each(base.tiles.begin(), base.tiles.end(), iterateTiles);

    for (int x = xSearchRange.start; x <= xSearchRange.end; x++) {
        for (int y = ySearchRange.start; y < ySearchRange.end; y++) {
            if (x % multipleOf != 0 && y % multipleOf != 0) {
                continue;
            }

            doneIterations++;
            if (alignmentShift == nullptr) {
                auto align = Point2i(x, y);
                alignmentShift = &align;
            }
            double alignmentError = compare(base.layer, layer.layer, x + prevAlignmentX, y + prevAlignmentY);
            if (alignmentError < minAlignmentError) {
                minAlignmentError = alignmentError;
                alignmentShift->x = x + prevAlignmentX;
                alignmentShift->y = y + prevAlignmentY;
            }
        }
    }
    std::printf("processed %d iterations on layer=%d\n", doneIterations, base.number);
    std::printf("minAlignmentError=%f\n", minAlignmentError);
    cout << "alignmentShift: " << *alignmentShift << endl;
    std::printf("--------------------------\n");
    return *alignmentShift;
}

cv::Point2d ImageAlign::align(Mat &base, Mat &shifted) {
    auto start = std::chrono::high_resolution_clock::now();

    auto basePyr = gaussianPyramid(base);
    auto shiftedPyr = gaussianPyramid(shifted);
    assert(basePyr.size() == shiftedPyr.size());

    auto minSearchDistance = 24; // min search range 24 pixels
    auto xAlignment = 0.;
    auto yAlignment = 0.;
    auto xSearchRange = Range(-1 * minSearchDistance, 1 * minSearchDistance);
    auto ySearchRange = Range(-1 * minSearchDistance, 1 * minSearchDistance);

    auto initialAlignment = comparePyramidLayer(basePyr[basePyr.size() - 1],
                                                shiftedPyr[shiftedPyr.size() - 1],
                                                xSearchRange,
                                                ySearchRange,
                                                xAlignment,
                                                yAlignment);

    /* auto initialAlignment = comparePyramidLayerFFT(basePyr[basePyr.size() - 1],
                                                    shiftedPyr[shiftedPyr.size() - 1],
                                                    basePyr.size() - 1,
                                                    xAlignment,
                                                    yAlignment);*/


    xAlignment = initialAlignment.x;
    yAlignment = initialAlignment.y;


    for (int layer = basePyr.size() - 2; layer > 0; layer--) { // we do not aligning 0 level layers.. so skipping them.
        auto baseLayer = basePyr[layer];
        auto shiftedLayer = shiftedPyr[layer];
        xAlignment *= 2.;
        yAlignment *= 2.;
        auto scale = 1;
        xSearchRange.start = -1 * (minSearchDistance * scale);
        xSearchRange.end = 1 * (minSearchDistance * scale);
        ySearchRange.start = -1 * (minSearchDistance * scale);
        ySearchRange.end = 1 * (minSearchDistance * scale);
        Point2i alignOffset = comparePyramidLayer(baseLayer,
                                                  shiftedLayer,
                                                  xSearchRange,
                                                  ySearchRange,
                                                  xAlignment,
                                                  yAlignment);

        /*Point2d alignOffset = comparePyramidLayerFFT(baseLayer,
                                                     shiftedLayer,
                                                     layer,
                                                     xAlignment,
                                                     yAlignment);*/

        xAlignment = alignOffset.x;
        yAlignment = alignOffset.y;
    }

    xAlignment *= 2.;
    yAlignment *= 2.;

    /*auto finalShift = comparePyramidLayer(basePyr[0], shiftedPyr[0], 0, xSearchRange, ySearchRange, xAlignment,
                                          yAlignment);
    xAlignment = finalShift.x;
    yAlignment = finalShift.y;*/
    std::printf("finalAlignment=%fx%f\n", xAlignment, yAlignment);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::printf("duration=%ld.ms\n", duration.count() / 1000);
    return {xAlignment, yAlignment};
}

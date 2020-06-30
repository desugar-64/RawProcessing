#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/ImageAlign.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "input/ImageReader.h"
#include "merge/ImageMerge.h"

void mergeAveraged(cv::Mat &base, cv::Mat &aligned);

void applyMask(cv::Mat &mat);

using namespace cv;

void processRaw() {
    Demosaic demosaic("/home/sergeyfitis/raw/plane.dng", "/home/sergeyfitis/raw/plane_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
//    processRaw();
    std::cout << "Ok!" << std::endl;
//    Mat base = imread("../align_test/plane_base.jpg", IMREAD_GRAYSCALE);
//    Mat baseColor = imread("../align_test/plane_base.jpg", IMREAD_COLOR);
//    Mat shifted = imread("../align_test/plane_shifted.jpg", IMREAD_GRAYSCALE);
//    Mat shiftedColor = imread("../align_test/plane_shifted.jpg", IMREAD_COLOR);

    ImageAlign aligner;
    ImageMerge merger;

    std::vector<cv::Mat> shots = ImageReader::readFolder("../burst3", 16);
    assert(!shots.empty());
    auto &baseShot = shots[0]; // TODO: find the less blurry shot
    Mat baseGrey;
    cvtColor(baseShot, baseGrey, CV_BGR2GRAY);

    std::map<int, std::vector<Tile>> shotAlignments;
    Mat shotGrey(baseGrey.rows, baseGrey.cols, baseGrey.type());

    Mat noAlign = baseShot.clone();

    for (auto &shotShifted : shots) {
        addWeighted(noAlign, 0.5, shotShifted, 0.5, 0, noAlign);
    }

    imwrite("no_align.png", noAlign);

    for (int shotIdx = 1; shotIdx < shots.size(); shotIdx++) {
        printf("\n\n\n------------------------------------\n");
        printf("processing %d frame of %lu\n", shotIdx, shots.size() - 1);
        auto &shotShifted = shots[shotIdx];
        cvtColor(shotShifted, shotGrey, CV_BGR2GRAY);
        auto shotShiftOffsets = aligner.align(baseGrey, shotGrey);
        shotAlignments[shotIdx] = shotShiftOffsets;
    }

    printf("\n------------------------------------\n");
    Mat merged = merger.mergeBurst(baseShot, shots, shotAlignments);

    imwrite("aligned.png", merged);

    /*Mat noAlign;
    addWeighted(baseColor, 0.5, shiftedColor, 0.5, 0, noAlign);
    imwrite("no_align.png", noAlign);

    cvtColor(baseColor, baseColor, COLOR_BGR2BGRA);
    cvtColor(shiftedColor, shiftedColor, COLOR_BGR2BGRA);
    baseColor.convertTo(baseColor, CV_8UC4);
    shiftedColor.convertTo(shiftedColor, CV_8UC4);

    auto tilesOffsets = aligner.align(base, shifted);

    Mat aligned = baseColor.clone();

    for (auto &tile : tilesOffsets) {
        if (tile.alignmentError <= 3.) {
            auto tileMat = shiftedColor(tile.rect).clone();
            applyMask(tileMat);
            Mat shiftCorrectedTile =
                    ImageAlign::translateImg(tileMat, tile.alignmentOffset.x, tile.alignmentOffset.y);
            Mat alignTo = aligned(tile.rect);
            mergeAveraged(alignTo, shiftCorrectedTile);
        }
    }

    imwrite("align.png", aligned);

    Rect roi = Rect(2400, 2000, 400, 400);

    imwrite("align_.png", aligned(roi));
    imwrite("base_.png", baseColor(roi));*/
//    namedWindow("aligned", WINDOW_NORMAL);
//    imshow("aligned", aligned);


//    cv::waitKey(0);
    return 0;
}

void applyMask(Mat &mat) {

    const auto fileMask = imread("../tile_mask.png", IMREAD_GRAYSCALE);
//    namedWindow("f_mask", WINDOW_NORMAL);
//    imshow("f_mask", fileMask);
//    waitKey();
    fileMask.convertTo(fileMask, CV_8UC1);

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            auto &pixels = mat.at<Vec4b>(y, x);
            auto &maskPixel = fileMask.at<ushort>(y, x);
//            pixels[3] *= mask[y][x];
            pixels[3] *= (65535 - maskPixel) / 65535.;
        }
    }
//    imwrite("tiles/tile_a.png", mat);
//    namedWindow("mask", WINDOW_NORMAL);
//    imshow("mask", mat);
//    waitKey();
}

void mergeAveraged(Mat &base, Mat &aligned) {
    for (int y = 0; y < base.rows; y++) {
        for (int x = 0; x < base.cols; x++) {
            auto &alignedPixels = aligned.at<Vec4b>(y, x);
            auto &basePixels = base.at<Vec4b>(y, x);

            bool isTransparent = alignedPixels[3] != 255;

            basePixels[0] = isTransparent ? basePixels[0] : (basePixels[0] + alignedPixels[0]) / 2; // B
            basePixels[1] = isTransparent ? basePixels[1] : (basePixels[1] + alignedPixels[1]) / 2; // G
            basePixels[2] = isTransparent ? basePixels[2] : (basePixels[2] + alignedPixels[2]) / 2; // R
//            basePixels[2] = isTransparent ? basePixels[2] : (basePixels[2] + alignedPixels[2]); // R
            basePixels[3] = 255;                                                                    // A
        }
    }
}


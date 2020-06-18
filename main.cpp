#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/ImageAlign.h"

using namespace cv;

void processRaw() {
    Demosaic demosaic("/home/sergeyfitis/raw/day.dng", "/home/sergeyfitis/raw/day_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
//    processRaw();
    std::cout << "Ok!" << std::endl;
    Mat base = imread("../align_test/day_base.jpg", IMREAD_GRAYSCALE);
    Mat baseColor = imread("../align_test/day_base.jpg", IMREAD_COLOR);
    Mat shifted = imread("../align_test/day_shifted.jpg", IMREAD_GRAYSCALE);
    Mat shiftedColor = imread("../align_test/day_shifted.jpg", IMREAD_COLOR);

    Mat noAlign;
    addWeighted(baseColor, 0.5, shiftedColor, 0.5, 0, noAlign);
    imwrite("no_align.jpg", noAlign);

    cvtColor(baseColor, baseColor, COLOR_BGR2BGRA);
    cvtColor(shiftedColor, shiftedColor, COLOR_BGR2BGRA);
    baseColor.convertTo(baseColor, CV_8UC4);
    shiftedColor.convertTo(shiftedColor, CV_8UC4);

    ImageAlign aligner;
    auto tilesOffsets = aligner.align(base, shifted);

    Mat aligned = baseColor.clone();
    int tileIdx = 0;
    for (auto &tile : tilesOffsets) {
        if (tile.alignmentError <= 4.) {
            auto tileMat = shiftedColor(tile.rect).clone();
            Mat shiftCorrectedTile =
                    ImageAlign::translateImg(tileMat, tile.alignmentOffset.x, tile.alignmentOffset.y) / 2;
            Mat alignTo = aligned(tile.rect);
            addWeighted(alignTo, 0.5, shiftCorrectedTile, 0.5, 0.0, alignTo);
            if (tileIdx < 4000) {
                imwrite(format("tiles/tile_%d.png", tileIdx), alignTo);
            }
//            alignTo.copyTo(aligned(tile.rect));
            tileIdx++;
        }
    }

    imwrite("align.jpg", aligned);
//    namedWindow("aligned", WINDOW_NORMAL);
//    imshow("aligned", aligned);


//    cv::waitKey(0);
    return 0;
}


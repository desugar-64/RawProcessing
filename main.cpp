#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/ImageAlign.h"

using namespace cv;

void processRaw() {
    Demosaic demosaic("/home/sergeyfitis/raw/flower.dng", "/home/sergeyfitis/raw/flower_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
//    processRaw();
    std::cout << "Ok!" << std::endl;
    Mat base = imread("../align_test/flower_base.jpg", IMREAD_GRAYSCALE);
    Mat baseColor = imread("../align_test/flower_base.jpg", IMREAD_COLOR);
    Mat shifted = imread("../align_test/flower_shifted.jpg", IMREAD_GRAYSCALE);
    Mat shiftedColor = imread("../align_test/flower_shifted.jpg", IMREAD_COLOR);
    ImageAlign aligner;
    auto tilesOffsets = aligner.align(base, shifted);

    Mat aligned = baseColor.clone();
    for (auto &tile : tilesOffsets) {
        if (tile.alignmentError < 5.) {
            auto tileMat = shiftedColor(tile.rect);
            Mat shiftCorrectedTile =
                    ImageAlign::translateImg(tileMat, tile.alignmentOffset.x, tile.alignmentOffset.y) / 2;
            shiftCorrectedTile.copyTo(aligned(tile.rect));
        }
    }

//    namedWindow("aligned", WINDOW_NORMAL);
//    imshow("aligned", aligned);

    Mat noAlign;
    addWeighted(baseColor, 0.5, shiftedColor, 0.5, 0, noAlign);
    imwrite("no_align.jpg", noAlign);
    imwrite("align.jpg", aligned);

    cv::waitKey(0);
    return 0;
}


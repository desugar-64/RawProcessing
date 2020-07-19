#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui.hpp"
#include "iostream"
#include "raw/Demosaic.h"
#include "align/ImageAlign.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "input/ImageReader.h"
#include "merge/ImageMerge.h"

using namespace cv;

void processRaw() {
    // Demosaic demosaic("/home/sergeyfitis/raw/plane.dng", "/home/sergeyfitis/raw/plane_orig.jpg");
    // demosaic.generateRGBComponents();
    // demosaic.interpolateGRBG();
    // demosaic.colorize();
    // demosaic.squaredDifference();
    // demosaic.display();
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

    std::vector<cv::Mat> shots = ImageReader::readFolder("burst3", 3);
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



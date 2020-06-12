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

Mat translateImg(Mat &img, int xOffset, int yOffset) {
    Mat out;
    Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
    warpAffine(img, out, trans_mat, img.size());

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
//    diff = abs(base - compare);
//    namedWindow("diff", WINDOW_NORMAL);
//    imshow("diff", diff);
    auto mean = cv::mean(diff);
    return sum(mean).val[0];
}

static Point2i comparePyramidLayer(Mat &base, Mat &layer, int layerNumber, Range xSearchRange, Range ySearchRange) {
    Point2i *alignShift = nullptr;
    double minAlignOffset = std::numeric_limits<double>::max();
    for (int x = xSearchRange.start; x <= xSearchRange.end; x++) {
        for (int y = ySearchRange.start; y < ySearchRange.end; y++) {
            if (alignShift == nullptr) {
                auto align = Point2i(x, y);
                alignShift = &align;
            }
            double alignOffset = compare(base, layer, x, y);
            if (alignOffset < minAlignOffset) {
                minAlignOffset = alignOffset;
                alignShift->x = x;
                alignShift->y = y;
            }
        }
    }
    std::printf("minAlignOffset=%f\n", minAlignOffset);
    std::printf("alignOffset=%dx%d\n", alignShift->x, alignShift->y);
    return *alignShift;
}

//
// Created by sergeyfitis on 03.06.20.
//

#include "Demosaic.h"

#include "cmath"
#include "libraw/libraw.h"
#include "opencv2/highgui/highgui_c.h"

using namespace std;
using namespace cv;


Demosaic::Demosaic(const char *filePath, const char *originalImgPath) {
    assert(libRaw.open_file(filePath) == LIBRAW_SUCCESS);
    assert(libRaw.unpack() == LIBRAW_SUCCESS);
    assert(libRaw.raw2image_ex(1) == LIBRAW_SUCCESS);
//    assert(libRaw.dcraw_process() == LIBRAW_SUCCESS);

    int width = libRaw.imgdata.rawdata.sizes.iwidth;
    int height = libRaw.imgdata.rawdata.sizes.iheight;

    Mat raw(height, width, CV_16UC1, libRaw.imgdata.rawdata.raw_image);
    raw.convertTo(image, CV_32F);

    rows = image.rows;
    cols = image.cols;

    r = Mat::zeros(rows, cols, CV_32F);
    g = Mat::zeros(rows, cols, CV_32F);
    b = Mat::zeros(rows, cols, CV_32F);

    demosaicImage = Mat::zeros(rows, cols, CV_32FC3);
    result = Mat::zeros(rows, cols, CV_32FC3);
    colorImage = imread(originalImgPath);
    colorImage.convertTo(colorImage, CV_32F);
}

/**
 * Split bayer (GRBG) mosaic image into separate R, G, and B matrices
 * G R G
 * B G B
 * G R G
 */
void Demosaic::generateRGBComponents() {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (y % 2 == 0) { // even rows, GRGRGR
                if (x % 2 == 0) {
                    // G Component
                    g.at<float>(y, x) = image.at<float>(y, x);
                } else {
                    // R Component
                    r.at<float>(y, x) = image.at<float>(y, x);
                }
            } else { // odd rows, BGBGBG
                if (x % 2 == 0) {
                    // B Component
                    b.at<float>(y, x) = image.at<float>(y, x);
                } else {
                    // B Component
                    g.at<float>(y, x) = image.at<float>(y, x);
                }
            }
        }
    }

//    cv::namedWindow("r", CV_WINDOW_FREERATIO);
//    cv::namedWindow("g", CV_WINDOW_FREERATIO);
//    cv::namedWindow("b", CV_WINDOW_FREERATIO);
//    Mat r_, g_, b_;
//    r.convertTo(r_, CV_8UC1);
//    g.convertTo(g_, CV_8UC1);
//    b.convertTo(b_, CV_8UC1);
//
//    imwrite("bayer_r.jpg", r_);
//    imwrite("bayer_b.jpg", b_);
//    imwrite("bayer_g.jpg", g_);
//
//    imshow("r", r_);
//    imshow("g", g_);
//    imshow("b", b_);
}

/**
 * Interpolate the missing information for each of the R, G, and B component matrices
 * G R G
 * B G B
 * G R G
 */
void Demosaic::interpolateGRBG() {

    // G component
    float k4[] = {
            1. / 5, 0., 1. / 5,
            0., 1. / 5, 0.,
            1. / 5, 0., 1. / 5
    };

    // R component
    float k1[] = {
            0., 1. / 2, 0.,
            0., 0., 0.,
            0., 1. / 2, 0.
    };

    // B component
    float k2[] = {
            0., 0., 0.,
            1. / 2, 0., 1. / 2,
            0., 0., 0.
    };

    float k3[]{
            1. / 4., 0., 1. / 4.,
            0., 0., 0.,
            1. / 4., 0., 1. / 4.
    };

    Mat kernel1 = Mat(3, 3, CV_32F, k1);
    Mat kernel2 = Mat(3, 3, CV_32F, k2);
    Mat kernel3 = Mat(3, 3, CV_32F, k3);
    Mat kernel4 = Mat(3, 3, CV_32F, k4);

    Mat r2 = Mat(rows, cols, CV_32F);
    Mat r3 = Mat(rows, cols, CV_32F);
    Mat r4 = Mat(rows, cols, CV_32F);

    Mat b2 = Mat(rows, cols, CV_32F);
    Mat b3 = Mat(rows, cols, CV_32F);
    Mat b4 = Mat(rows, cols, CV_32F);

    Mat g2 = Mat(rows, cols, CV_32F);
    Mat g3 = Mat(rows, cols, CV_32F);
    Mat g4 = Mat(rows, cols, CV_32F);

    filter2D(r, r2, -1, kernel1);
    filter2D(r, r3, -1, kernel2);
    filter2D(r, r4, -1, kernel3);
    Mat r_ = r2 + r3 + r4;
    r = (r + r_);


    filter2D(b, b2, -1, kernel1);
    filter2D(b, b3, -1, kernel2);
    filter2D(b, b4, -1, kernel3);
    Mat b_ = b2 + b3 + b4;
    b = (b + b_);
//    filter2D(g, g2, -1, kernel1);
    filter2D(g, g3, -1, kernel2);
//    filter2D(g, g4, -1, kernel4);
    g = (g + g2 + g3 + g4) / 2.;

}

//G R
//B G
void Demosaic::interpolateSimple() {


    // G component
    float gdata[] = {
            1 / 2., 0.,
            0., 1 / 2.
    };

    // R component
    float rdata[] = {
            0., 1.,
            0., 0.
    };

    // B component
    float bdata[] = {
            0., 0.,
            1., 0.
    };

    Mat rKernel = Mat(2, 2, CV_32F, rdata);
    Mat gKernel = Mat(2, 2, CV_32F, gdata);
    Mat bKernel = Mat(2, 2, CV_32F, bdata);

    Mat r2_ = Mat(rows, cols, CV_32F);
    Mat r3_ = Mat(rows, cols, CV_32F);
    Mat r4_ = Mat(rows, cols, CV_32F);

    Mat b2_ = Mat(rows, cols, CV_32F);
    Mat b3_ = Mat(rows, cols, CV_32F);
    Mat b4_ = Mat(rows, cols, CV_32F);

    Mat g2_ = Mat(rows, cols, CV_32F);
    Mat g3_ = Mat(rows, cols, CV_32F);
    Mat g4_ = Mat(rows, cols, CV_32F);

    filter2D(r, r2_, -1, rKernel);
    filter2D(r, r3_, -1, bKernel);
    filter2D(r, r4_, -1, gKernel);
    r = r + r2_ + r3_ + r4_;

    filter2D(g, g2_, -1, gKernel);
    filter2D(g, g3_, -1, rKernel);
    filter2D(g, g4_, -1, bKernel);
    g = (g + g2_ + g3_ + g4_) * 1 / 4.;

    filter2D(b, b2_, -1, bKernel);
    filter2D(b, b3_, -1, rKernel);
    filter2D(b, b4_, -1, gKernel);
    b = b + b2_ + b3_ + b4_;
}

/**
 * Combine R, G, and B component matrices into a single three channel color image
 */
void Demosaic::colorize() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            demosaicImage.at<Vec3f>(i, j) = Vec3f(
                    b.at<float>(i, j),
                    g.at<float>(i, j),
                    r.at<float>(i, j)
            );
        }
    }
}

/**
 * Highlight the artifacts produced by the demosaicing process
 */
void Demosaic::squaredDifference() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
                result.at<Vec3f>(i, j)[k] =
                        sqrt(pow((colorImage.at<Vec3f>(i, j)[k] - demosaicImage.at<Vec3f>(i, j)[k]), 2));
            }
        }
    }
}

/**
 * Implement proposed bilinear interpolation improvement
 */
void Demosaic::modifiedInterpolation() {
    Mat r_g = Mat(rows, cols, CV_32F);
    Mat b_g = Mat(rows, cols, CV_32F);

    r_g = r - g;
    b_g = b - g;

    medianBlur(r_g, r_g, 3);
    medianBlur(b_g, b_g, 3);

    r = r_g + g;
    b = b_g + g;
}

/**
 * Display results of the demosaicing process
 */
void Demosaic::display() {
    Mat d8u, r8u, c8u;
    demosaicImage.convertTo(d8u, CV_8UC3);
    imwrite("demosaic.jpg", d8u);
    result.convertTo(r8u, CV_8UC3);

    colorImage.convertTo(c8u, CV_8UC3);
    cv::namedWindow("Original", CV_WINDOW_FREERATIO);
    imshow("Original", c8u);
    cv::namedWindow("Squared Difference", CV_WINDOW_FREERATIO);
    imshow("Squared Difference", r8u);

    cv::namedWindow("Demosaic", CV_WINDOW_FREERATIO);
    imshow("Demosaic", d8u(Rect(0, 0, 4000, 3000)));

    waitKey(0);
}

// private


std::string Demosaic::getCfaPatternString(LibRaw *processor) {
    static const std::unordered_map<char, char> CDESC_TO_CFA = {
            {'R', 0},
            {'G', 1},
            {'B', 2},
            {'r', 0},
            {'g', 1},
            {'b', 2}
    };
    const auto &cdesc = processor->imgdata.idata.cdesc;
    return {
            CDESC_TO_CFA.at(cdesc[processor->COLOR(0, 0)]),
            CDESC_TO_CFA.at(cdesc[processor->COLOR(0, 1)]),
            CDESC_TO_CFA.at(cdesc[processor->COLOR(1, 0)]),
            CDESC_TO_CFA.at(cdesc[processor->COLOR(1, 1)])
    };
}

CfaPattern Demosaic::getCfaPattern(LibRaw *processor) {
    const auto cfa_pattern = getCfaPatternString(processor);
    if (cfa_pattern == std::string{0, 1, 1, 2}) {
        return CfaPattern::CFA_RGGB;
    } else if (cfa_pattern == std::string{1, 0, 2, 1}) {
        return CfaPattern::CFA_GRBG;
    } else if (cfa_pattern == std::string{2, 1, 1, 0}) {
        return CfaPattern::CFA_BGGR;
    } else if (cfa_pattern == std::string{1, 2, 0, 1}) {
        return CfaPattern::CFA_GBRG;
    }
//    throw std::invalid_argument("Unsupported CFA pattern: " + cfa_pattern);
    return CfaPattern::CFA_UNKNOWN;
}

Mat Demosaic::getColorCorrectionMatrix(LibRaw *processor) {
    const auto raw_color = processor->imgdata.color;
    auto ccm = Mat(3, 3, CV_32FC1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ccm.at<float>(i, j) = raw_color.rgb_cam[j][i];
        }
    }
    return ccm;
}
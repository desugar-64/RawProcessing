//
// Created by sergeyfitis on 03.06.20.
//

#include "Demosaic.h"

#include "cmath"
#include "libraw/libraw.h"
#include "opencv2/highgui/highgui_c.h"
#include "filter/FilterFactory.h"

using namespace std;
using namespace cv;


Demosaic::Demosaic(const char *filePath, const char *originalImgPath) {
    libRaw = new LibRaw;
    assert(libRaw->open_file(filePath) == LIBRAW_SUCCESS);
    assert(libRaw->unpack() == LIBRAW_SUCCESS);
    assert(libRaw->raw2image_ex(1) == LIBRAW_SUCCESS);
    ushort *rawImage = libRaw->imgdata.rawdata.raw_image;


    int width = libRaw->imgdata.rawdata.sizes.iwidth;
    int height = libRaw->imgdata.rawdata.sizes.iheight;

    auto cfaPattern = getCfaPattern(libRaw);
    bayerFilter = createFilter(cfaPattern, rawImage, width, height);

    int rows = bayerFilter->rows;
    int cols = bayerFilter->cols;
    demosaicImage = Mat::zeros(rows, cols, CV_32FC3);
    result = Mat::zeros(rows, cols, CV_32FC3);
    colorImage = imread(originalImgPath);
    colorImage.convertTo(colorImage, CV_32F);
}

void Demosaic::generateRGBComponents() {
    bayerFilter->generateComponents();
}

void Demosaic::interpolateGRBG() {
    bayerFilter->interpolate();
}

/**
 * Combine R, G, and B component matrices into a single three channel color image
 */
void Demosaic::colorize() {
    int rows = bayerFilter->rows;
    int cols = bayerFilter->cols;

    auto r = bayerFilter->bayerGreen;
    auto g = bayerFilter->bayerGreen;
    auto b = bayerFilter->bayerBlue;
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
    int rows = bayerFilter->rows;
    int cols = bayerFilter->cols;

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
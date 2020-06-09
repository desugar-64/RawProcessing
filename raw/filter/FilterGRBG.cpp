//
// Created by sergeyfitis on 10.06.20.
//

#include "Filter.h"
#include "FilterGRBG.h"

using namespace cv;

FilterGRBG::FilterGRBG(ushort *rawPixels, int width, int height) {
    Mat raw = Mat(height, width, CV_16UC1, rawPixels);
    raw.convertTo(rawImage, CV_32F);
    raw.release();

    rows = rawImage.rows;
    cols = rawImage.cols;

    bayerRed = Mat::zeros(rows, cols, CV_32F);
    bayerGreen = Mat::zeros(rows, cols, CV_32F);
    bayerBlue = Mat::zeros(rows, cols, CV_32F);
}

FilterGRBG::~FilterGRBG() {
    bayerRed.release();
    bayerGreen.release();
    bayerBlue.release();
}

/**
 * Split bayer GRBG mosaic rawImage into separate R, G, and B matrices
 * G R G
 * B G B
 * G R G
 */
void FilterGRBG::generateComponents() {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (y % 2 == 0) { // even rows, GRGRGR
                if (x % 2 == 0) {
                    // G Component
                    bayerGreen.at<float>(y, x) = rawImage.at<float>(y, x);
                } else {
                    // R Component
                    bayerRed.at<float>(y, x) = rawImage.at<float>(y, x);
                }
            } else { // odd rows, BGBGBG
                if (x % 2 == 0) {
                    // B Component
                    bayerBlue.at<float>(y, x) = rawImage.at<float>(y, x);
                } else {
                    // B Component
                    bayerGreen.at<float>(y, x) = rawImage.at<float>(y, x);
                }
            }
        }
    }
}


/**
* Interpolate the missing information for each of the R, G, and B component matrices
* G R G
* B G B
* G R G
*/
void FilterGRBG::interpolate() {
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

    filter2D(bayerRed, r2, -1, kernel1);
    write("bayerRed_k1", bayerRed);
    filter2D(bayerRed, r3, -1, kernel2);
    write("bayerRed_k2", bayerRed);
    filter2D(bayerRed, r4, -1, kernel3);
    write("bayerRed_k2", bayerRed);
    bayerRed = bayerRed + r2 + r3 + r4;
    write("bayerRed_all", bayerRed);


    filter2D(bayerBlue, b2, -1, kernel1);
    write("bayerBlue_k1", bayerBlue);
    filter2D(bayerBlue, b3, -1, kernel2);
    write("bayerBlue_k2", bayerBlue);
    filter2D(bayerBlue, b4, -1, kernel3);
    write("bayerBlue_k3", bayerBlue);
    bayerBlue = bayerBlue + b2 + b3 + b4;
    write("bayerBlue_all", bayerBlue);

//    filter2D(bayerGreen, g2, -1, kernel4);
//    filter2D(bayerGreen, g3, -1, kernel1);
    filter2D(bayerGreen, g4, -1, kernel2);
    write("bayerGreen_k2", bayerGreen);
    bayerGreen = (bayerGreen + g2 + g3 + g4)/2.;
    write("bayerGreen_all", bayerGreen);
}

void FilterGRBG::write(const char *name, const Mat& img) {
    Mat i;
    img.convertTo(i, CV_8UC1);
    imwrite(format("%s.jpg", name), i);
}
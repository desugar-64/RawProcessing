//
// Created by sergeyfitis on 28.06.20.
//

#include "ImageReader.h"

using namespace std;
using namespace cv;

std::vector<cv::Mat> ImageReader::readFolder(const string &path, int filesToRead) {
    vector<Mat> files = vector<Mat>();
    for (int i = 0; i < filesToRead; i++) {
        auto mat = imread(format("%s/%d.jpg", path.c_str(), i));
        assert(!mat.empty());
        cvtColor(mat, mat, COLOR_BGR2BGRA);
        mat.convertTo(mat, CV_8UC4);
        files.push_back(mat);
    }
    return files;
}

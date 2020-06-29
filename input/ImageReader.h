//
// Created by sergeyfitis on 28.06.20.
//

#ifndef RAWPROCESSING_IMAGEREADER_H
#define RAWPROCESSING_IMAGEREADER_H

#include "opencv2/opencv.hpp"

class ImageReader {
public:
    static std::vector<cv::Mat> readFolder(const std::string &path, int filesToRead);
};


#endif //RAWPROCESSING_IMAGEREADER_H

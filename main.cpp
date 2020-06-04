#include "iostream"
#include "raw/Demosaic.h"

using namespace cv;

void processRaw() {
    Demosaic demosaic("~/raw/flower.dng", "~/raw/flower_orig.jpg");
    demosaic.generateRGBComponents();
    demosaic.interpolateGRBG();
//        demosaic.modifiedInterpolation();
//        demosaic.interpolateSimple();
    demosaic.colorize();
    demosaic.squaredDifference();
    demosaic.display();
}

int main() {
    processRaw();
    std::cout << "Ok!" << std::endl;
    return 0;
}


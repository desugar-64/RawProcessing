#include "iostream"
#include "raw/Demosaic.h"

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
    processRaw();
    std::cout << "Ok!" << std::endl;
    return 0;
}


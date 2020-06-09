//
// Created by sergeyfitis on 10.06.20.
//

#include "Filter.h"
#include "FilterGRBG.h"

static Filter* createFilter(CfaPattern pattern, ushort *rawPixels, int width, int height) {
    Filter* filter;
    switch (pattern) {
        case CFA_RGGB:
            break;
        case CFA_GRBG:
            filter = new FilterGRBG(rawPixels, width, height);
            break;
        case CFA_BGGR:
            break;
        case CFA_GBRG:
            break;
        case CFA_UNKNOWN:
            filter = nullptr;
            break;
    }
    return filter;
}
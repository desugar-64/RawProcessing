//
// Created by sergeyfitis on 24.06.20.
//

#include "ImageMerge.h"
#include "../align/ImageAlign.h"

using namespace cv;

void ImageMerge::merge(Mat &merged, Mat &base, Mat &align, std::vector<Tile> &alignmentOffsets, uchar numberOfImages) {
    for (auto &tile : alignmentOffsets) {
        if (tile.alignmentError <= 3.) {
            auto tileMat = align(tile.rect);
            auto baseTileMat = base(tile.rect);
//            applyMask(tileMat);
            Mat shiftCorrectedTile =
                    ImageAlign::translateImg(tileMat, tile.alignmentOffset.x, tile.alignmentOffset.y);
            Mat mergeTo = merged(tile.rect);
            mergeAveraged(mergeTo, baseTileMat, shiftCorrectedTile, numberOfImages);
        }
    }
}

void ImageMerge::mergeAveraged(Mat &mergeTo, Mat &base, Mat &align, uchar numberOfImages) {
    for (int y = 0; y < base.rows; y++) {
        for (int x = 0; x < base.cols; x++) {
            auto &alignedPixels = align.at<Vec4b>(y, x);
            auto &basePixels = base.at<Vec4b>(y, x);
            auto &mergePixels = mergeTo.at<Vec4b>(y, x);
            bool isTransparent = alignedPixels[3] != 255;

            int b = basePixels[0] + alignedPixels[0];
            int g = basePixels[1] + alignedPixels[1];
            int r = basePixels[2] + alignedPixels[2];

            mergePixels[0] = isTransparent ? basePixels[0] : b / 2;       // B
            mergePixels[1] = isTransparent ? basePixels[1] : g / 2;       // G

            if (DEBUG_TILES) {
                mergePixels[2] = 255;         // R
            } else {
                mergePixels[2] = isTransparent ? basePixels[2] : r / 2;   // R
            }

            mergePixels[3] = 255;                                                      // A
        }
    }
}

cv::Mat
ImageMerge::mergeBurst(Mat &base, std::vector<cv::Mat> burst, std::map<int, std::vector<Tile>> &alignmentOffsets) {
    Mat merged = base.clone();
    for (int shotIdx = 1; shotIdx < burst.size(); shotIdx++) {
        printf("merging %d frame of %lu\n", shotIdx, burst.size() - 1);

        auto &shot = burst[shotIdx];
        auto &tiles = alignmentOffsets[shotIdx];
        for (auto &tile : tiles) {
            if (tile.alignmentError <= ALIGNMENT_PRECISION) {
                auto misalignedTileMat = shot(tile.rect);
                auto mergeTileMatCopy = merged(tile.rect).clone();
                auto mergeTileMat = merged(tile.rect);
                Mat shiftCorrectedTile =
                        ImageAlign::translateImg(misalignedTileMat, tile.alignmentOffset.x, tile.alignmentOffset.y);
                mergeAveraged(mergeTileMat, mergeTileMatCopy, shiftCorrectedTile, burst.size());
            }
        }
    }
    return merged;
}

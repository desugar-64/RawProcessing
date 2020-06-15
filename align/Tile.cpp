//
// Created by sergeyfitis on 15.06.20.
//
#include "Tile.h"

Tile::Tile(cv::Rect2i &rect, cv::Rect2i &alignmentOffset, double alignmentError) {
    this->rect = rect;
    this->alignmentOffset = alignmentOffset;
    this->alignmentError = alignmentError;
}
//
// Created by sergeyfitis on 16.06.20.
//

#include "PyramidLayer.h"
#include "ImageAlign.h"

using namespace cv;
using namespace std;

PyramidLayer::PyramidLayer(Mat &layer, int number) {
    this->layer = layer;
    this->number = number;
}

void PyramidLayer::generateTiles() {
    CV_Assert(!layer.empty());
    CV_Assert(layer.channels() == 1);


    int width = layer.cols;
    int height = layer.rows;

    printf("generate tiles on layer %d, WxH=%dx%d\n", number, width, height);

    vector<Tile> layerTiles;

    double tileDownscale = 1.;
    for (int l = 1; l <= number; l++) {
        tileDownscale *= 2;
    }
    int tileSize = floor((T_SIZE / tileDownscale));
    printf("tile size %d, downscale %f\n", tileSize, tileDownscale);

    int tileCountX = width / tileSize;
    int tileCountY = height / tileSize;
    printf("tile count on axis x=%d, y=%d\n", tileCountX, tileCountY);
    printf("-----------------------------------\n");

    for (int y = 0; y <= height * 2; y++) {
        for (int x = 0; x <= width * 2; x++) {

            if ((x % tileSize == 0 && y % tileSize == 0) || (x == 0 && y == 0)) {
                int xTileOffset = x / 2;
                int yTileOffset = y / 2;

                int tileWidth = min(tileSize, abs(width - xTileOffset));
                int tileHeight = min(tileSize, abs(height - yTileOffset));

                if (tileHeight > 1 && tileWidth > 1) {
                    auto rect = Rect2i(xTileOffset, yTileOffset, tileWidth, tileHeight);
                    auto tile = Tile(rect);
                    layerTiles.push_back(tile);
                }

            }
        }
    }
// Visual debugging...
//    for (int i = 0; i < layerTiles.size(); i++) {
//        auto tile = layerTiles[i];
//        rectangle(layer, tile.rect, Scalar(255, 255, 255), 1);
//    }
//
//    auto winName = format("layer %d", number);
//    namedWindow(winName, WINDOW_NORMAL);
//    imshow(winName, layer);
//    Mat lastTile = layer(layerTiles[layerTiles.size() - 1].rect);
//    Mat firstTile = layer(layerTiles[0].rect);
//    namedWindow("firstTile", WINDOW_NORMAL);
//    namedWindow("lastTile", WINDOW_NORMAL);
//    imshow("lastTile", lastTile);
//    imshow("firstTile", firstTile);
//    waitKey(0);

    this->tiles = layerTiles;
}

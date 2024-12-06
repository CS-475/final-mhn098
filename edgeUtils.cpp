#include "edgeUtils.h"

Edge clipEdge(const Edge& edge, int canvasWidth, int canvasHeight) {
    Edge clippedEdge = edge;

    // Clip top
    if (clippedEdge.top < 0) {
        clippedEdge.xLeft = (int)(std::round(clippedEdge.m * -clippedEdge.top + clippedEdge.b));
        clippedEdge.top = 0;
    }

    // Clip bottom 
    if (clippedEdge.bottom > canvasHeight) {
        clippedEdge.bottom = canvasHeight;
    }

    // Clip left 
    if (clippedEdge.xLeft < 0) {
        clippedEdge.xLeft = 0;
        clippedEdge.top = (int)(std::round((0 - clippedEdge.b) / clippedEdge.m));
    }

    // Clip right 
    if (clippedEdge.xRight > canvasWidth) {
        clippedEdge.xRight = canvasWidth;
        clippedEdge.bottom = (int)(std::round((canvasWidth - clippedEdge.b) / clippedEdge.m));
    }

    return clippedEdge;
}

Edge makeEdge(const GPoint& p0, const GPoint& p1) {
    GPoint top = p0, bottom = p1;
    int winding = 1;
    if (p0.y > p1.y) {
        std::swap(top, bottom);
        winding = -1;
    }

    Edge edge;
    edge.m = (bottom.x - top.x) / (bottom.y - top.y);
    edge.b = top.x - edge.m * top.y;
    edge.top = GRoundToInt(top.y);
    edge.bottom = GRoundToInt(bottom.y);
    edge.currX = GRoundToInt(top.x);
    edge.winding = winding;
    return edge;
}
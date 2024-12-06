#ifndef EDGE_UTILS_H
#define EDGE_UTILS_H

#include "include/GPoint.h"
#include <cmath>

struct Edge {
    float m;    // Slope of the edge
    float b;    // Y-intercept of the edge
    int top;    // Top Y-coordinate (rounded)
    int bottom; // Bottom Y-coordinate (rounded)
    int xLeft;  // Left X-coordinate at current scanline
    int xRight; // Right X-coordinate at current scanline
    int currX;  // X-coordinate at the current scanline
    int winding; // +1 for upward edges, -1 for downward edges

    bool isValid(int y) const {
        return (y >= top && y < bottom); 
    }

    float computeX(int y) const {
        return m * y + b; 
    }

    bool isUseful() const {
        return top < bottom;
    }
};

Edge clipEdge(const Edge& edge, int canvasWidth, int canvasHeight);
Edge makeEdge(const GPoint& p0, const GPoint& p1);

#endif // EDGE_UTILS_H
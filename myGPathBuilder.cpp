#include "include/GPathBuilder.h"
#include "include/GMatrix.h"
#include <cmath>

inline void GPathBuilder::reset() {
    fPts.clear(); 
    fVbs.clear(); 
}

void GPathBuilder::addRect(const GRect& rect, GPathDirection dir) {
    GPoint topLeft = {rect.left, rect.top};
    GPoint topRight = {rect.right, rect.top};
    GPoint bottomRight = {rect.right, rect.bottom};
    GPoint bottomLeft = {rect.left, rect.bottom};

    if (dir == GPathDirection::kCW) {
        moveTo(topLeft);
        lineTo(topRight);
        lineTo(bottomRight);
        lineTo(bottomLeft);
        lineTo(topLeft); 
    } else { 
        moveTo(topLeft);
        lineTo(bottomLeft);
        lineTo(bottomRight);
        lineTo(topRight);
        lineTo(topLeft); 
    }
}

void GPathBuilder::addPolygon(const GPoint pts[], int count) {
    assert(count > 0); 
    moveTo(pts[0]); 
    for (int i = 1; i < count; ++i) {
        lineTo(pts[i]);
    }
}

constexpr float kCircleConstant = 0.551915;  // The best value for that length seems to be approximately 0.551915

void GPathBuilder::addCircle(GPoint center, float radius, GPathDirection dir) {
    float offset = kCircleConstant * radius;

    GPoint srcPoints[12] = {
        {1, 0},                // Right
        {1, offset},           // Bottom-right control
        {offset, 1},           // Bottom-right
        {0, 1},                // Bottom
        {-offset, 1},          // Bottom-left control
        {-1, offset},          // Bottom-left
        {-1, 0},               // Left
        {-1, -offset},         // Top-left control
        {-offset, -1},         // Top-left
        {0, -1},               // Top
        {offset, -1},          // Top-right control
        {1, -offset}           // Top-right
    };

    GPoint dstPoints[12];

    GMatrix matrix = GMatrix::Translate(center.x, center.y) * GMatrix::Scale(radius, radius);
    matrix.mapPoints(dstPoints, srcPoints, 12);

    int indices[12];
    if (dir == GPathDirection::kCW) {
        indices[0] = 0; indices[1] = 1; indices[2] = 2;
        indices[3] = 3; indices[4] = 4; indices[5] = 5;
        indices[6] = 6; indices[7] = 7; indices[8] = 8;
        indices[9] = 9; indices[10] = 10; indices[11] = 11;
    } else {
        indices[0] = 0; indices[1] = 11; indices[2] = 10;
        indices[3] = 9; indices[4] = 8; indices[5] = 7;
        indices[6] = 6; indices[7] = 5; indices[8] = 4;
        indices[9] = 3; indices[10] = 2; indices[11] = 1;
    }

    moveTo(dstPoints[indices[0]]);

    for (int i = 0; i < 12; i += 3) {
        quadTo(dstPoints[indices[i + 1]], dstPoints[indices[i + 2]]);
    }
}


inline void GPathBuilder::transform(const GMatrix& matrix) {
    for (size_t i = 0; i < fPts.size(); ++i) {
        fPts[i] = matrix * fPts[i];
    }
}

inline std::shared_ptr<GPath> GPathBuilder::detach() {
    auto path = std::make_shared<GPath>(std::move(fPts), std::move(fVbs));
    reset();
    return path;
}
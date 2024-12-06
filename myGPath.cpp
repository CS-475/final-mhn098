#include "include/GPath.h"
#include "include/GMatrix.h"
#include <algorithm> 
#include "include/GRect.h"

void updateQuadBounds(const GPoint pts[3], float& minX, float& minY, float& maxX, float& maxY) {
    const float epsilon = 1e-6f;

    minX = std::min({minX, pts[0].x, pts[2].x});
    minY = std::min({minY, pts[0].y, pts[2].y});
    maxX = std::max({maxX, pts[0].x, pts[2].x});
    maxY = std::max({maxY, pts[0].y, pts[2].y});

    float d = pts[0].x - 2 * pts[1].x + pts[2].x;
    if (std::abs(d) > epsilon) {
        float tx = (pts[0].x - pts[1].x) / d;
        if (tx >= 0 && tx <= 1) {
            float x = (1 - tx) * (1 - tx) * pts[0].x + 2 * (1 - tx) * tx * pts[1].x + tx * tx * pts[2].x;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
        }
    }

    d = pts[0].y - 2 * pts[1].y + pts[2].y;
    if (std::abs(d) > epsilon) {
        float ty = (pts[0].y - pts[1].y) / d;
        if (ty >= 0 && ty <= 1) {
            float y = (1 - ty) * (1 - ty) * pts[0].y + 2 * (1 - ty) * ty * pts[1].y + ty * ty * pts[2].y;
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }
}


void updateCubicBounds(const GPoint pts[4], float& minX, float& minY, float& maxX, float& maxY) {
    const float epsilon = 1e-6f;

    minX = std::min({minX, pts[0].x, pts[3].x});
    minY = std::min({minY, pts[0].y, pts[3].y});
    maxX = std::max({maxX, pts[0].x, pts[3].x});
    maxY = std::max({maxY, pts[0].y, pts[3].y});

    float ax = -pts[0].x + 3 * pts[1].x - 3 * pts[2].x + pts[3].x;
    float bx = 3 * pts[0].x - 6 * pts[1].x + 3 * pts[2].x;
    float cx = -3 * pts[0].x + 3 * pts[1].x;

    std::vector<float> tValues;
    float discriminant = bx * bx - 4 * ax * cx;
    if (discriminant >= 0 && std::abs(ax) > epsilon) {
        float sqrtD = sqrt(discriminant);
        tValues.push_back((-bx + sqrtD) / (2 * ax));
        tValues.push_back((-bx - sqrtD) / (2 * ax));
    }

    for (float t : tValues) {
        if (t >= 0 && t <= 1) {
            float x = (1 - t) * (1 - t) * (1 - t) * pts[0].x +
                      3 * (1 - t) * (1 - t) * t * pts[1].x +
                      3 * (1 - t) * t * t * pts[2].x +
                      t * t * t * pts[3].x;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
        }
    }

    // Repeat for y
    float ay = -pts[0].y + 3 * pts[1].y - 3 * pts[2].y + pts[3].y;
    float by = 3 * pts[0].y - 6 * pts[1].y + 3 * pts[2].y;
    float cy = -3 * pts[0].y + 3 * pts[1].y;

    discriminant = by * by - 4 * ay * cy;
    if (discriminant >= 0 && std::abs(ay) > epsilon) {
        float sqrtD = sqrt(discriminant);
        tValues.clear();
        tValues.push_back((-by + sqrtD) / (2 * ay));
        tValues.push_back((-by - sqrtD) / (2 * ay));
    }

    for (float t : tValues) {
        if (t >= 0 && t <= 1) {
            float y = (1 - t) * (1 - t) * (1 - t) * pts[0].y +
                      3 * (1 - t) * (1 - t) * t * pts[1].y +
                      3 * (1 - t) * t * t * pts[2].y +
                      t * t * t * pts[3].y;
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }
}

GRect GPath::bounds() const {
    if (fPts.empty()) {
        return GRect{0, 0, 0, 0};
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();

    GPoint pts[4];

    Iter iter(*this);
    while (auto verbOpt = iter.next(pts)) {
        GPathVerb verb = *verbOpt;
        switch (verb) {
            case GPathVerb::kMove:
                minX = std::min(minX, pts[0].x);
                minY = std::min(minY, pts[0].y);
                maxX = std::max(maxX, pts[0].x);
                maxY = std::max(maxY, pts[0].y);
                break;

            case GPathVerb::kLine:
                minX = std::min({minX, pts[0].x, pts[1].x});
                minY = std::min({minY, pts[0].y, pts[1].y});
                maxX = std::max({maxX, pts[0].x, pts[1].x});
                maxY = std::max({maxY, pts[0].y, pts[1].y});
                break;

            case GPathVerb::kQuad:
                updateQuadBounds(pts, minX, minY, maxX, maxY);
                break;

            case GPathVerb::kCubic:
                updateCubicBounds(pts, minX, minY, maxX, maxY);
                break;

            default:
                break;
        }
    }

    return GRect{minX, minY, maxX, maxY};
}

void GPath::ChopQuadAt(const GPoint src[3], GPoint dst[5], float t) {
    GPoint p01 = (1 - t) * src[0] + t * src[1]; // Point between P0 and P1
    GPoint p12 = (1 - t) * src[1] + t * src[2]; // Point between P1 and P2
    GPoint p012 = (1 - t) * p01 + t * p12;      // The new split point on the curve
    
    dst[0] = src[0];
    dst[1] = p01;
    dst[2] = p012;
    
    dst[3] = p12;
    dst[4] = src[2];
}

void GPath::ChopCubicAt(const GPoint src[4], GPoint dst[7], float t) {
    GPoint p01 = (1 - t) * src[0] + t * src[1]; 
    GPoint p12 = (1 - t) * src[1] + t * src[2]; 
    GPoint p23 = (1 - t) * src[2] + t * src[3];
    
    GPoint p012 = (1 - t) * p01 + t * p12; 
    GPoint p123 = (1 - t) * p12 + t * p23; 
    
    GPoint p0123 = (1 - t) * p012 + t * p123; 
    
    dst[0] = src[0];
    dst[1] = p01;
    dst[2] = p012;
    dst[3] = p0123;
    
    dst[4] = p123;
    dst[5] = p23;
    dst[6] = src[3];
}


inline std::shared_ptr<GPath> GPath::transform(const GMatrix& matrix) const {
    std::vector<GPoint> transformedPts(fPts.size());
    matrix.mapPoints(transformedPts.data(), fPts.data(), fPts.size());
    return std::make_shared<GPath>(std::move(transformedPts), fVbs);
}

inline GPath::Iter::Iter(const GPath& path)
    : fCurrPt(path.fPts.data())
    , fCurrVb(path.fVbs.data())
    , fStopVb(path.fVbs.data() + path.fVbs.size())
{}

inline nonstd::optional<GPathVerb> GPath::Iter::next(GPoint pts[]) {
    if (fCurrVb < fStopVb) {
        GPathVerb verb = *fCurrVb++;
        switch (verb) {
            case kMove:
                pts[0] = *fCurrPt++;
                return verb;
            case kLine:
                pts[0] = *(fCurrPt++);
                pts[1] = *(fCurrPt++);
                return verb;
            case kQuad:
                pts[0] = *(fCurrPt++);
                pts[1] = *(fCurrPt++);
                pts[2] = *(fCurrPt++);
                return verb;
            case kCubic:
                pts[0] = *(fCurrPt++);
                pts[1] = *(fCurrPt++);
                pts[2] = *(fCurrPt++);
                pts[3] = *(fCurrPt++);
                return verb;
            default:
                break;
        }
    }
    return nonstd::nullopt;
}

inline GPath::Edger::Edger(const GPath& path)
    : fPrevMove(nullptr)
    , fCurrPt(path.fPts.data())
    , fCurrVb(path.fVbs.data())
    , fStopVb(path.fVbs.data() + path.fVbs.size())
    , fPrevVerb(-1)
{}

inline nonstd::optional<GPathVerb> GPath::Edger::next(GPoint pts[]) {
    while (fCurrVb < fStopVb) {
        GPathVerb verb = *fCurrVb++;
        fPrevVerb = static_cast<int>(verb);
        
        switch (verb) {
            case kMove:
                fPrevMove = fCurrPt; 
                pts[0] = *fCurrPt++;
                return verb;
            case kLine: {
                if (fPrevMove) { 
                    pts[0] = *fPrevMove;
                    pts[1] = *fCurrPt++;
                    return verb;
                }
                break;
            }
            default:
                break;
        }
    }
    return nonstd::nullopt; 
}
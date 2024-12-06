#include "include/GCanvas.h"
#include "include/GPaint.h"
#include "include/GRect.h"
#include "include/GPixel.h"
#include "include/GMatrix.h"
#include "include/GPath.h"
#include "include/GPathBuilder.h"
#include "myGCanvas.h"
#include "blendUtils.h"
#include "edgeUtils.h"
#include "myGShader.h"
#include <iostream>

// Utility function for fast division by 255
static inline int GDiv255(int value) {
    return (value + 128) * 257 >> 16;
}

// Inline function to draw a triangle with optional colors and textures
void drawTriangleInline(const GPoint verts[3], const GColor colors[3], const GPoint texs[3], const GPaint& paint, const GBitmap& device);

// Constructor for MyCanvas class
MyCanvas::MyCanvas(GBitmap& bitmap) : fBitmap(bitmap) {}

// Destructor for MyCanvas class
MyCanvas::~MyCanvas() {}

// Converts GColor to GPixel (ARGB format)
GPixel MyCanvas::colorToPixel(const GColor& color) const {
    uint8_t a = static_cast<uint8_t>(color.a * 255 + 0.5f);
    uint8_t r = static_cast<uint8_t>(color.r * color.a * 255 + 0.5f);
    uint8_t g = static_cast<uint8_t>(color.g * color.a * 255 + 0.5f);
    uint8_t b = static_cast<uint8_t>(color.b * color.a * 255 + 0.5f);
    return GPixel_PackARGB(a, r, g, b);
}

// Clears the canvas with a solid color
void MyCanvas::clear(const GColor& color) {
    int height = fBitmap.height();
    int width = fBitmap.width();
    GPixel p = colorToPixel(color);
    GPixel *row_addr = nullptr;

    // Loop through each row
    for (int y = 0; y < height; y++) {
        row_addr = fBitmap.getAddr(0, y);

        // Loop through each pixel in the row
        for (int x = 0; x < width; x++) {
            row_addr[x] = p; // Set the pixel to the color
        }
    }
}

// Blends two pixels based on the specified blend mode
GPixel MyCanvas::blendPixel(GPixel dstPixel, GPixel srcPixel, GBlendMode blendMode) const {
    BlendProc blendFunc = findBlend(blendMode); // Get the blending function
    return blendFunc(srcPixel, dstPixel);       // Apply blending
}

// Draws a rectangle using four points
void MyCanvas::drawRect(const GRect& rect, const GPaint& paint) {
    GPoint points[4] = {
        {rect.left, rect.top},
        {rect.right, rect.top},
        {rect.right, rect.bottom},
        {rect.left, rect.bottom}
    };
    drawConvexPolygon(points, 4, paint); // Delegate to convex polygon drawing
}

// Draws a convex polygon with specified points
void MyCanvas::drawConvexPolygon(const GPoint points[], int count, const GPaint& paint) {
    GPoint transformedPoints[count];

    // Apply transformation to each point
    for (int i = 0; i < count; ++i) {
        float sx = points[i].x;
        float sy = points[i].y;
        transformedPoints[i].x = fMatrix[0] * sx + fMatrix[2] * sy + fMatrix[4];
        transformedPoints[i].y = fMatrix[1] * sx + fMatrix[3] * sy + fMatrix[5];
    }

    std::vector<Edge> edges;

    // Create edges from points
    for (int i = 0; i < count; ++i) {
        int next = (i + 1) % count; // Wrap around to form a closed shape
        Edge edge = makeEdge(transformedPoints[i], transformedPoints[next]);
        if (edge.isUseful()) {
            edges.push_back(edge);
        }
    }

    // Sort edges by their top coordinate
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.top < b.top;
    });

    int canvasHeight = fBitmap.height();
    GShader* shader = paint.peekShader();
    bool useShader = (shader != nullptr && shader->setContext(fMatrix));
    GPixel srcPixel;

    // Loop through each scanline
    for (int y = 0; y < canvasHeight; ++y) {
        std::vector<int> intersections;

        // Find all intersections of edges with the current scanline
        for (const auto& edge : edges) {
            if (y >= edge.top && y <= edge.bottom) {
                int x = static_cast<int>(std::round(edge.m * y + edge.b));
                intersections.push_back(x);
            }
        }

        // Sort the intersections to determine spans
        std::sort(intersections.begin(), intersections.end());

        if (useShader) {
            shader->shadeRow(0, y, 1, &srcPixel);
        } else {
            srcPixel = colorToPixel(paint.getColor());
        }

        GPixel* rowAddr = fBitmap.getAddr(0, y);

        // Loop through each span between intersections
        for (size_t i = 0; i < intersections.size(); i += 2) {
            if (i + 1 < intersections.size()) {
                int xStart = std::max(0, intersections[i]);
                int xEnd = std::min(fBitmap.width(), intersections[i + 1]);

                if (xStart < xEnd) {
                    // Fill the span
                    for (int x = xStart; x < xEnd; ++x) {
                        if (useShader) {
                            shader->shadeRow(x, y, 1, &srcPixel);
                        }
                        rowAddr[x] = blendPixel(rowAddr[x], srcPixel, paint.getBlendMode());
                    }
                }
            }
        }
    }
}

// Pushes the current transformation matrix onto the stack
void MyCanvas::save() {
    fMatrixStack.push(fMatrix);
}

// Restores the last saved transformation matrix
void MyCanvas::restore() {
    if (!fMatrixStack.empty()) {
        fMatrix = fMatrixStack.top();
        fMatrixStack.pop();
    }
}

// Concatenates a new transformation matrix with the current matrix
void MyCanvas::concat(const GMatrix& matrix) {
    fMatrix = GMatrix::Concat(fMatrix, matrix);
}
// Computes the number of segments required for a quadratic curve approximation
int computeQuadSegments(const GPoint pts[3], float tolerance) {
    float ax = pts[0].x - 2 * pts[1].x + pts[2].x;
    float ay = pts[0].y - 2 * pts[1].y + pts[2].y;
    float maxDist = std::sqrt(ax * ax + ay * ay);
    return static_cast<int>(std::ceil(std::sqrt(maxDist / tolerance)));
}

// Computes the number of segments required for a cubic curve approximation
int computeCubicSegments(const GPoint pts[4], float tolerance) {
    float ax = -pts[0].x + 3 * (pts[1].x - pts[2].x) + pts[3].x;
    float ay = -pts[0].y + 3 * (pts[1].y - pts[2].y) + pts[3].y;
    float maxDist = std::sqrt(ax * ax + ay * ay);
    return static_cast<int>(std::ceil(std::sqrt(maxDist / tolerance)));
}

// Evaluates a point on a quadratic Bezier curve for a given parameter t
GPoint evalQuad(const GPoint pts[3], float t) {
    float u = 1 - t; // Complement of t
    return {
        u * u * pts[0].x + 2 * u * t * pts[1].x + t * t * pts[2].x,
        u * u * pts[0].y + 2 * u * t * pts[1].y + t * t * pts[2].y
    };
}

// Evaluates a point on a cubic Bezier curve for a given parameter t
GPoint evalCubic(const GPoint pts[4], float t) {
    float u = 1 - t; // Complement of t
    return {
        u * u * u * pts[0].x + 3 * u * u * t * pts[1].x + 3 * u * t * t * pts[2].x + t * t * t * pts[3].x,
        u * u * u * pts[0].y + 3 * u * u * t * pts[1].y + 3 * u * t * t * pts[2].y + t * t * t * pts[3].y
    };
}

// Draws a path on the canvas using the specified paint
void MyCanvas::drawPath(const GPath& path, const GPaint& paint) {
    GPathBuilder builder;
    builder.reset(); // Clear any previous path
    GPath::Iter iter(path); // Iterator for the path
    GPoint pts[GPath::kMaxNextPoints];
    const float tolerance = 0.25f; // Tolerance for curve approximations

    // Iterate through the path segments
    while (auto verb = iter.next(pts)) {
        switch (verb.value()) {
            case GPathVerb::kMove:
                builder.moveTo(pts[0]); // Start a new sub-path
                break;
            case GPathVerb::kLine:
                builder.lineTo(pts[1]); // Add a straight line
                break;
            case GPathVerb::kQuad: {
                int segments = computeQuadSegments(pts, tolerance);
                for (int i = 0; i < segments; ++i) {
                    float t1 = static_cast<float>(i + 1) / segments;
                    builder.lineTo(evalQuad(pts, t1)); // Approximate the curve
                }
                break;
            }
            case GPathVerb::kCubic: {
                int segments = computeCubicSegments(pts, tolerance);
                for (int i = 0; i < segments; ++i) {
                    float t1 = static_cast<float>(i + 1) / segments;
                    builder.lineTo(evalCubic(pts, t1)); // Approximate the curve
                }
                break;
            }
            default:
                break; // Ignore unsupported segments
        }
    }

    // Transform and render the path
    builder.transform(fMatrix);
    auto transformedPath = builder.detach();
    GPath::Edger edger(*transformedPath); // Edge iterator
    std::vector<Edge> edges;

    // Convert path segments to edges
    while (auto verb = edger.next(pts)) {
        if (verb.value() == GPathVerb::kLine) {
            Edge edge = makeEdge(pts[0], pts[1]);
            if (edge.isUseful()) edges.push_back(edge);
        }
    }

    // Sort edges by their top coordinate
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.top < b.top || (a.top == b.top && a.currX < b.currX);
    });

    int canvasHeight = fBitmap.height();
    int canvasWidth = fBitmap.width();
    GShader* shader = paint.peekShader();
    bool useShader = (shader != nullptr && shader->setContext(fMatrix));
    GPixel srcPixel;

    // Loop through each scanline
    for (int y = 0; y < canvasHeight; ++y) {
        std::vector<int> xIntervals;
        int winding = 0;

        // Find intersections for the current scanline
        for (auto& edge : edges) {
            if (edge.isValid(y)) {
                int x = static_cast<int>(std::round(edge.computeX(y)));
                if (x < 0 || x >= canvasWidth) continue;
                xIntervals.push_back(x);
                winding += edge.winding;
            }
        }

        // Sort intersections to determine spans
        std::sort(xIntervals.begin(), xIntervals.end());

        // Fill spans between intersections
        for (size_t i = 0; i + 1 < xIntervals.size(); i += 2) {
            int left = xIntervals[i];
            int right = xIntervals[i + 1];
            left = std::max(0, left);
            right = std::min(canvasWidth, right);

            for (int xSpan = left; xSpan < right; ++xSpan) {
                if (useShader) shader->shadeRow(xSpan, y, 1, &srcPixel);
                else srcPixel = colorToPixel(paint.getColor());
                GPixel* dst = fBitmap.getAddr(xSpan, y);
                *dst = blendPixel(*dst, srcPixel, paint.getBlendMode());
            }
        }
    }
}

// Factory function to create a canvas instance
std::unique_ptr<GCanvas> GCreateCanvas(const GBitmap& bitmap) {
    return std::make_unique<MyCanvas>(const_cast<GBitmap&>(bitmap));
}

// Draws a star with a gradient fill on the canvas
std::string GDrawSomething(GCanvas* canvas, GISize dim) {
    canvas->clear(GColor::RGBA(0.8f, 0.8f, 0.8f, 1.0f)); // Clear canvas with a light gray color

    float centerX = dim.width / 2.0f;
    float centerY = dim.height / 2.0f;
    float outerRadius = 100.0f;
    float innerRadius = 50.0f;

    // Define gradient start and end points
    GPoint gradientStart = {centerX, centerY - outerRadius};
    GPoint gradientEnd = {centerX, centerY + outerRadius};

    // Define gradient colors
    GColor colors[] = {
        GColor::RGBA(1.0f, 0.0f, 0.0f, 1.0f), // Red
        GColor::RGBA(0.0f, 1.0f, 0.0f, 1.0f), // Green
        GColor::RGBA(0.0f, 0.0f, 1.0f, 1.0f)  // Blue
    };

    // Create a gradient shader
    std::shared_ptr<GShader> gradientShader = GCreateLinearGradient(
        gradientStart, gradientEnd, colors, 3);

    GPaint gradientPaint;
    gradientPaint.setShader(gradientShader);

    // Create points for a 10-pointed star
    const int pointsCount = 10;
    GPoint starPoints[pointsCount];
    for (int i = 0; i < pointsCount; ++i) {
        float angle = i * M_PI / 5.0; // Divide circle into 10 parts
        float radius = (i % 2 == 0) ? outerRadius : innerRadius; // Alternate radius
        starPoints[i] = {
            centerX + radius * std::cos(angle),
            centerY - radius * std::sin(angle)
        };
    }

    canvas->drawConvexPolygon(starPoints, pointsCount, gradientPaint);
    return "Star!";
}

// Draws a mesh defined by vertices, colors, and optional texture coordinates
void MyCanvas::drawMesh(const GPoint verts[], const GColor colors[], const GPoint texs[], 
                        int count, const int indices[], const GPaint& paint) {
    // If no shader is provided, disable texture coordinates
    if (!paint.peekShader()) {
        texs = nullptr;
    }
    // If neither colors nor textures are provided, nothing to draw
    if (!colors && !texs) {
        return;
    }

    // Loop through each triangle defined by indices
    for (int i = 0; i < count; ++i) {
        int p0 = indices[3 * i];
        int p1 = indices[3 * i + 1];
        int p2 = indices[3 * i + 2];

        // Prepare the triangle vertices
        GPoint transformedVerts[3];
        GPoint originalVerts[3] = {verts[p0], verts[p1], verts[p2]};
        fMatrix.mapPoints(transformedVerts, originalVerts, 3);

        // Prepare triangle colors if available
        GColor triColors[3];
        if (colors) {
            triColors[0] = colors[p0];
            triColors[1] = colors[p1];
            triColors[2] = colors[p2];
        }

        // Prepare triangle texture coordinates if available
        GPoint triTexs[3];
        if (texs) {
            triTexs[0] = texs[p0];
            triTexs[1] = texs[p1];
            triTexs[2] = texs[p2];
        }

        // Draw the triangle
        drawTriangleInline(transformedVerts, 
                           colors ? triColors : nullptr,
                           texs ? triTexs : nullptr,
                           paint, fBitmap);
    }
}

// Draws a subdivided quadrilateral
void MyCanvas::drawQuad(const GPoint verts[4], const GColor colors[4], const GPoint texs[4], 
                        int level, const GPaint& paint) {
    // If level is invalid or vertices are null, do nothing
    if (level < 1 || !verts) {
        return;
    }
    // If no colors or textures are provided, nothing to draw
    if (!colors && !texs) {
        return;
    }

    try {
        std::vector<GPoint> generatedVerts;   // Generated vertices for the subdivided quad
        std::vector<GColor> generatedColors; // Interpolated colors for vertices
        std::vector<GPoint> generatedTexs;   // Interpolated texture coordinates
        std::vector<int> indices;            // Indices defining triangles

        // Compute the number of vertices for the grid
        int numVerts = (level + 1) * (level + 1);

        // Generate vertices, colors, and textures for the subdivided grid
        for (int i = 0; i <= level; ++i) {
            float v = static_cast<float>(i) / level;
            for (int j = 0; j <= level; ++j) {
                float u = static_cast<float>(j) / level;

                // Bilinearly interpolate vertex positions
                float x = verts[0].x * (1 - u) * (1 - v) +
                          verts[1].x * u * (1 - v) +
                          verts[3].x * (1 - u) * v +
                          verts[2].x * u * v;
                float y = verts[0].y * (1 - u) * (1 - v) +
                          verts[1].y * u * (1 - v) +
                          verts[3].y * (1 - u) * v +
                          verts[2].y * u * v;
                generatedVerts.push_back({x, y});

                // Interpolate colors if provided
                if (colors) {
                    float c00 = (1 - u) * (1 - v);
                    float c10 = u * (1 - v);
                    float c01 = (1 - u) * v;
                    float c11 = u * v;

                    GColor interpolatedColor;
                    interpolatedColor.a = c00 * colors[0].a + c10 * colors[1].a + c01 * colors[3].a + c11 * colors[2].a;
                    interpolatedColor.r = c00 * colors[0].r + c10 * colors[1].r + c01 * colors[3].r + c11 * colors[2].r;
                    interpolatedColor.g = c00 * colors[0].g + c10 * colors[1].g + c01 * colors[3].g + c11 * colors[2].g;
                    interpolatedColor.b = c00 * colors[0].b + c10 * colors[1].b + c01 * colors[3].b + c11 * colors[2].b;

                    generatedColors.push_back(interpolatedColor);
                }

                // Interpolate texture coordinates if provided
                if (texs) {
                    float tx = (1 - u) * (1 - v) * texs[0].x +
                               u * (1 - v) * texs[1].x +
                               (1 - u) * v * texs[3].x +
                               u * v * texs[2].x;
                    float ty = (1 - u) * (1 - v) * texs[0].y +
                               u * (1 - v) * texs[1].y +
                               (1 - u) * v * texs[3].y +
                               u * v * texs[2].y;

                    generatedTexs.push_back({tx, ty});
                }
            }
        }

        // Generate triangle indices for the grid
        for (int i = 0; i < level; ++i) {
            for (int j = 0; j < level; ++j) {
                int idx = i * (level + 1) + j;

                // First triangle in the grid cell
                indices.push_back(idx);
                indices.push_back(idx + 1);
                indices.push_back(idx + level + 1);

                // Second triangle in the grid cell
                indices.push_back(idx + 1);
                indices.push_back(idx + level + 2);
                indices.push_back(idx + level + 1);
            }
        }

        // Draw the mesh using the generated data
        drawMesh(generatedVerts.data(),
                 colors ? generatedColors.data() : nullptr,
                 texs ? generatedTexs.data() : nullptr,
                 indices.size() / 3, indices.data(), paint);
    } catch (const std::exception& e) {
        // Handle any runtime exceptions
        std::cout << "ERROR in drawQuad: " << e.what() << std::endl;
    } catch (...) {
        // Handle unexpected errors
        std::cout << "UNKNOWN ERROR in drawQuad" << std::endl;
    }
}

// Computes barycentric coordinates for a point relative to a triangle
void computeBarycentric(float x, float y, const GPoint& v0, const GPoint& v1, const GPoint& v2, 
                        float& alpha, float& beta, float& gamma) {
    float denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    alpha = ((v1.y - v2.y) * (x - v2.x) + (v2.x - v1.x) * (y - v2.y)) / denom;
    beta = ((v2.y - v0.y) * (x - v2.x) + (v0.x - v2.x) * (y - v2.y)) / denom;
    gamma = 1.0f - alpha - beta;
}

// Implements blending logic for pixels based on blending modes
GPixel blendPixel(const GPixel& dst, const GPixel& src, GBlendMode mode) {
    if (mode == GBlendMode::kSrc) {
        return src; // Source replaces destination
    }

    if (mode == GBlendMode::kSrcOver) {
        // Extract alpha components
        int srcA = GPixel_GetA(src);
        int dstA = GPixel_GetA(dst);

        // Compute output alpha
        int outA = srcA + GDiv255((255 - srcA) * dstA);

        // Compute output RGB components
        int outR = GPixel_GetR(src) + GDiv255((255 - srcA) * GPixel_GetR(dst));
        int outG = GPixel_GetG(src) + GDiv255((255 - srcA) * GPixel_GetG(dst));
        int outB = GPixel_GetB(src) + GDiv255((255 - srcA) * GPixel_GetB(dst));

        return GPixel_PackARGB(outA, outR, outG, outB);
    }

    // Fallback to source pixel for unsupported blend modes
    return src;
}

// Draws a triangle on the canvas with optional colors and textures
void drawTriangleInline(const GPoint verts[3], const GColor colors[3], const GPoint texs[3], 
                        const GPaint& paint, const GBitmap& device) {
    // Determine bounding box of the triangle
    int left = std::max(0, static_cast<int>(std::floor(std::min({verts[0].x, verts[1].x, verts[2].x}))));
    int right = std::min(device.width(), static_cast<int>(std::ceil(std::max({verts[0].x, verts[1].x, verts[2].x}))));
    int top = std::max(0, static_cast<int>(std::floor(std::min({verts[0].y, verts[1].y, verts[2].y}))));
    int bottom = std::min(device.height(), static_cast<int>(std::ceil(std::max({verts[0].y, verts[1].y, verts[2].y}))));

    // Compute edge deltas for barycentric coordinates
    float dx12 = verts[1].x - verts[0].x;
    float dy12 = verts[1].y - verts[0].y;
    float dx23 = verts[2].x - verts[1].x;
    float dy23 = verts[2].y - verts[1].y;
    float dx31 = verts[0].x - verts[2].x;
    float dy31 = verts[0].y - verts[2].y;

    GShader* shader = paint.peekShader();
    bool useShader = shader && shader->setContext(GMatrix());

    float px = 0.5f; // Pixel center adjustment
    float py = 0.5f;

    // Loop through each scanline in the bounding box
    for (int y = top; y < bottom; ++y) {
        GPixel* row = device.getAddr(0, y); // Get row address in the bitmap
        float fy = y + py; // Adjust y for pixel center

        // Loop through each pixel in the row
        for (int x = left; x < right; ++x) {
            float fx = x + px; // Adjust x for pixel center

            // Compute barycentric weights
            float w0 = dx23 * (fy - verts[1].y) - dy23 * (fx - verts[1].x);
            float w1 = dx31 * (fy - verts[2].y) - dy31 * (fx - verts[2].x);
            float w2 = dx12 * (fy - verts[0].y) - dy12 * (fx - verts[0].x);

            // Calculate triangle area for normalization
            float area = dx23 * dy31 - dx31 * dy23;

            // Skip pixels if the area is zero (degenerate triangle)
            if (area == 0) { continue; }

            float invArea = 1.0f / area;
            w0 *= invArea;
            w1 *= invArea;
            w2 *= invArea;

            // Check if the pixel is inside the triangle
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                GPixel srcPixel;

                // Use the shader if textures are available
                if (useShader && texs) {
                    float tx = w0 * texs[0].x + w1 * texs[1].x + w2 * texs[2].x;
                    float ty = w0 * texs[0].y + w1 * texs[1].y + w2 * texs[2].y;
                    shader->shadeRow(tx, ty, 1, &srcPixel);
                } 
                // Otherwise, interpolate colors if available
                else if (colors) {
                    float a = w0 * colors[0].a + w1 * colors[1].a + w2 * colors[2].a;
                    float r = w0 * colors[0].r + w1 * colors[1].r + w2 * colors[2].r;
                    float g = w0 * colors[0].g + w1 * colors[1].g + w2 * colors[2].g;
                    float b = w0 * colors[0].b + w1 * colors[1].b + w2 * colors[2].b;

                    // Clamp values to ensure valid color components
                    a = std::max(0.0f, std::min(1.0f, a));
                    r = std::max(0.0f, std::min(1.0f, r)) * a;
                    g = std::max(0.0f, std::min(1.0f, g)) * a;
                    b = std::max(0.0f, std::min(1.0f, b)) * a;

                    srcPixel = GPixel_PackARGB(
                        static_cast<uint8_t>(a * 255 + 0.5f),
                        static_cast<uint8_t>(r * 255 + 0.5f),
                        static_cast<uint8_t>(g * 255 + 0.5f),
                        static_cast<uint8_t>(b * 255 + 0.5f)
                    );
                }

                // Blend the source pixel with the destination pixel
                row[x] = blendPixel(row[x], srcPixel, paint.getBlendMode());
            }
        }
    }
}
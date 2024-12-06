#include "include/GShader.h"
#include "include/GBitmap.h"
#include "include/GMatrix.h"
#include <memory>
#include <optional>
#include <algorithm>
#include <cmath>

class BitmapShader : public GShader {
public:
    BitmapShader(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode tileMode)
        : fBitmap(bitmap), fLocalMatrix(localMatrix), fTileMode(tileMode) {}

    bool isOpaque() override {
        for (int y = 0; y < fBitmap.height(); ++y) {
            for (int x = 0; x < fBitmap.width(); ++x) {
                if (GPixel_GetA(*fBitmap.getAddr(x, y)) != 255) {
                    return false;  
                }
            }
        }
        return true;  
    }

    bool setContext(const GMatrix& ctm) override {
        GMatrix combined = GMatrix::Concat(ctm, fLocalMatrix);
        fInverse = combined.invert();
        return fInverse.has_value(); 
    }

    void shadeRow(int x, int y, int count, GPixel row[]) override {
        GPoint srcPoint;
        GPoint dstPoint;

        for (int i = 0; i < count; ++i) {
            dstPoint = {x + i + 0.5f, y + 0.5f};
            fInverse->mapPoints(&srcPoint, &dstPoint, 1);

            float srcX = srcPoint.x;
            float srcY = srcPoint.y;

            // Apply tile mode
            srcX = applyTileMode(srcX, fBitmap.width(), fTileMode);
            srcY = applyTileMode(srcY, fBitmap.height(), fTileMode);

            int clampedX = std::clamp(static_cast<int>(srcX), 0, fBitmap.width() - 1);
            int clampedY = std::clamp(static_cast<int>(srcY), 0, fBitmap.height() - 1);

            row[i] = *fBitmap.getAddr(clampedX, clampedY);
        }
    }

private:
    GBitmap fBitmap;
    GMatrix fLocalMatrix;
    nonstd::optional<GMatrix> fInverse; 
    GTileMode fTileMode;

    float applyTileMode(float coord, int limit, GTileMode mode) const {
        switch (mode) {
            case GTileMode::kClamp:
                return std::clamp(coord, 0.0f, static_cast<float>(limit - 1));
            case GTileMode::kRepeat:
                return std::fmod(coord, limit);
            case GTileMode::kMirror:
                int wholePart = static_cast<int>(std::floor(coord / limit));
                float fracPart = coord - wholePart * limit;
                return (wholePart % 2 == 0) ? fracPart : limit - 1 - fracPart;
        }
        return coord; // Should not reach here.
    }
};

std::shared_ptr<GShader> GCreateBitmapShader(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode tileMode) {
    if (bitmap.width() <= 0 || bitmap.height() <= 0) {
        return nullptr; 
    }
    return std::make_shared<BitmapShader>(bitmap, localMatrix, tileMode);
}


class LinearGradientShader : public GShader {
public:
    LinearGradientShader(GPoint p0, GPoint p1, const GColor colors[], int count, GTileMode tileMode)
        : fP0(p0), fP1(p1), fColors(colors, colors + count), fTileMode(tileMode) {}

    bool isOpaque() override {
        for (const auto& color : fColors) {
            if (color.a < 1.0f) {
                return false;
            }
        }
        return true;
    }

    bool setContext(const GMatrix& ctm) override {
        float dx = fP1.x - fP0.x;
        float dy = fP1.y - fP0.y;
        float D = sqrt(dx * dx + dy * dy);

        if (D == 0) {
            return false;
        }

        GMatrix gradientMatrix(dx, -dy, 0, dy, dx, 0);
        GMatrix translationMatrix(1, 0, fP0.x, 0, 1, fP0.y);
        GMatrix combinedMatrix = GMatrix::Concat(ctm, GMatrix::Concat(translationMatrix, gradientMatrix));
        
        auto inv = combinedMatrix.invert();  
        if (!inv.has_value()) {
            return false;
        }
        fLocalMatrix = inv.value();
        return true;
    }

    void shadeRow(int x, int y, int count, GPixel row[]) override {
        for (int i = 0; i < count; ++i) {
            GPoint devicePoint = {x + i + 0.5f, y + 0.5f};
            GPoint localPoint;
            fLocalMatrix.mapPoints(&localPoint, &devicePoint, 1);

            float t = applyTileMode(localPoint.x, fTileMode);

            GColor color = interpolateColor(t);
            row[i] = colorToPixel(color);
        }
    }

private:
    GPoint fP0, fP1;
    std::vector<GColor> fColors;
    GMatrix fLocalMatrix;
    GTileMode fTileMode;

    GColor interpolateColor(float t) const {
        int n = fColors.size();
        int idx = static_cast<int>(t * (n - 1));
        float localT = (t * (n - 1)) - idx;

        if (idx >= n - 1) {
            return fColors[n - 1];
        }

        GColor c0 = fColors[idx];
        GColor c1 = fColors[idx + 1];

        return {
            c0.r * (1 - localT) + c1.r * localT,
            c0.g * (1 - localT) + c1.g * localT,
            c0.b * (1 - localT) + c1.b * localT,
            c0.a * (1 - localT) + c1.a * localT
        };
    }

    GPixel colorToPixel(const GColor& color) const {
        uint8_t a = static_cast<uint8_t>(color.a * 255 + 0.5f);
        uint8_t r = static_cast<uint8_t>(color.r * color.a * 255 + 0.5f);
        uint8_t g = static_cast<uint8_t>(color.g * color.a * 255 + 0.5f);
        uint8_t b = static_cast<uint8_t>(color.b * color.a * 255 + 0.5f);
        return GPixel_PackARGB(a, r, g, b);
    }

    float applyTileMode(float t, GTileMode mode) const {
        switch (mode) {
            case GTileMode::kClamp:
                return std::clamp(t, 0.0f, 1.0f);
            case GTileMode::kRepeat:
                return t - std::floor(t);
            case GTileMode::kMirror:
                int wholePart = static_cast<int>(std::floor(t));
                float fracPart = t - wholePart;
                return (wholePart % 2 == 0) ? fracPart : 1.0f - fracPart;
        }
        return t; // Should not reach here.
    }
};

std::shared_ptr<GShader> GCreateLinearGradient(GPoint p0, GPoint p1, const GColor colors[], int count, GTileMode tileMode) {
    if (count < 2) {
        return nullptr; 
    }
    return std::make_shared<LinearGradientShader>(p0, p1, colors, count, tileMode);
}
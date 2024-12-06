#ifndef MY_G_SHADER_H
#define MY_G_SHADER_H

#include "include/GShader.h"
#include "include/GBitmap.h"
#include "include/GMatrix.h"
#include <memory>
#include <optional>

class MyShader : public GShader {
public:
    MyShader(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode tileMode);

    bool isOpaque() override;
    bool setContext(const GMatrix& matrix) override;
    void shadeRow(int x, int y, int count, GPixel row[]) override;

private:
    GBitmap fDevice;
    GMatrix fLocalMatrix;
    std::optional<GMatrix> fInverse;
    GTileMode fTileMode;

    int applyTileMode(float coord, int size) const;
    float mirrored_t(float t) const;
};

std::shared_ptr<GShader> GCreateBitmapShader(const GBitmap& bitmap, const GMatrix& localMatrix, GTileMode mode);

#endif  // MY_G_SHADER_H
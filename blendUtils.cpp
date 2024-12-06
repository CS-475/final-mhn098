#include "blendUtils.h"

using BlendFunc = GPixel (*)(GPixel src, GPixel dst);

// function pointer type for blend operations
typedef GPixel (*BlendProc)(GPixel, GPixel);

// array of blend mode functions
const BlendProc gProcs[] = {
    clear_mode,    // GBlendMode::kClear
    src_mode,      // GBlendMode::kSrc
    dst_mode,      // GBlendMode::kDst
    src_over_mode, // GBlendMode::kSrcOver
    dst_over_mode, // GBlendMode::kDstOver
    src_in_mode,   // GBlendMode::kSrcIn
    dst_in_mode,   // GBlendMode::kDstIn
    src_out_mode,  // GBlendMode::kSrcOut
    dst_out_mode,  // GBlendMode::kDstOut
    src_atop_mode, // GBlendMode::kSrcATop
    dst_atop_mode, // GBlendMode::kDstATop
    xor_mode       // GBlendMode::kXor
};

BlendProc findBlend(GBlendMode mode) {
    switch (mode) {
        case GBlendMode::kClear: 
            return clear_mode;

        case GBlendMode::kSrc:
            return src_mode;

        case GBlendMode::kDst:
            return dst_mode;

        case GBlendMode::kSrcOver:
            return src_over_mode;

        case GBlendMode::kDstOver:
            return dst_over_mode;

        case GBlendMode::kSrcIn:
            return src_in_mode;

        case GBlendMode::kDstIn:
            return dst_in_mode;

        case GBlendMode::kSrcOut:
            return src_out_mode;

        case GBlendMode::kDstOut:
            return dst_out_mode;

        case GBlendMode::kSrcATop:
            return src_atop_mode;

        case GBlendMode::kDstATop:
            return dst_atop_mode;

        case GBlendMode::kXor:
            return xor_mode;
    }
    return nullptr;
}

GPixel clear_mode(GPixel src, GPixel dst) {
    return GPixel_PackARGB(0, 0, 0, 0);
}

GPixel src_mode(GPixel src, GPixel dst) {
    return src;
}

GPixel dst_mode(GPixel src, GPixel dst) {
    return dst;
}

GPixel src_over_mode(GPixel src, GPixel dst) {
    uint8_t srcA = GPixel_GetA(src);
    if (srcA == 0) {
        return dst;
    }
    if (srcA == 255) {
        return src;
    }
    uint8_t dstA = GPixel_GetA(dst);
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);

    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);

    uint8_t srcAlphaComp = 255 - srcA;

    uint8_t resultA = srcA + ((dstA * srcAlphaComp) >> 8);
    uint8_t resultR = srcR + ((dstR * srcAlphaComp) >> 8);
    uint8_t resultG = srcG + ((dstG * srcAlphaComp) >> 8);
    uint8_t resultB = srcB + ((dstB * srcAlphaComp) >> 8);
    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel dst_over_mode(GPixel src, GPixel dst) {
   uint8_t dstA = GPixel_GetA(dst);
    if (dstA == 0){
        return src;
    }
    if (dstA == 255){
        return dst;
    }

    uint8_t srcA = GPixel_GetA(src);
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);

    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);

    uint8_t dstAlphaComp = 255 - dstA;

    uint8_t resultA = dstA + ((srcA * dstAlphaComp) >> 8);
    uint8_t resultR = dstR + ((srcR * dstAlphaComp) >> 8);
    uint8_t resultG = dstG + ((srcG * dstAlphaComp) >> 8);
    uint8_t resultB = dstB + ((srcB * dstAlphaComp) >> 8);
    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel src_in_mode(GPixel src, GPixel dst) {
    uint8_t srcA = GPixel_GetA(src);
    uint8_t dstA = GPixel_GetA(dst);

    if (srcA == 0 || dstA == 0) {
        return GPixel_PackARGB(0, 0, 0, 0);
    }
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);
    uint8_t resultA = (srcA * dstA) >> 8;
    uint8_t resultR = (srcR * dstA) >> 8;
    uint8_t resultG = (srcG * dstA) >> 8;
    uint8_t resultB = (srcB * dstA) >> 8;

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel dst_in_mode(GPixel src, GPixel dst) {
     uint8_t srcA = GPixel_GetA(src);
    uint8_t dstA = GPixel_GetA(dst);

    if (srcA == 0 || dstA == 0) {
        return GPixel_PackARGB(0, 0, 0, 0);
    }
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);
    
    uint8_t resultA = (dstA * srcA) >> 8;
    uint8_t resultR = (dstR * srcA) >> 8;
    uint8_t resultG = (dstG * srcA) >> 8;
    uint8_t resultB = (dstB * srcA) >> 8;

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel src_out_mode(GPixel src, GPixel dst) {
     uint8_t srcA = GPixel_GetA(src);
    if (srcA == 0){
        return GPixel_PackARGB(0, 0, 0, 0);
    }
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);
    uint8_t dstA = GPixel_GetA(dst);

    uint8_t dstAlphaComp = 255 - dstA;

    uint8_t resultA = (srcA * dstAlphaComp) >> 8;
    uint8_t resultR = (srcR * dstAlphaComp) >> 8;
    uint8_t resultG = (srcG * dstAlphaComp) >> 8;
    uint8_t resultB = (srcB * dstAlphaComp) >> 8;

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel dst_out_mode(GPixel src, GPixel dst) {
    uint8_t dstA = GPixel_GetA(dst);
    if (dstA == 0){
        return GPixel_PackARGB(0, 0, 0, 0);
    }
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);
    uint8_t srcA = GPixel_GetA(src);

    uint8_t srcAlphaComp = 255 - srcA;

    uint8_t resultA = (dstA * srcAlphaComp) >> 8;
    uint8_t resultR = (dstR * srcAlphaComp) >> 8;
    uint8_t resultG = (dstG * srcAlphaComp) >> 8;
    uint8_t resultB = (dstB * srcAlphaComp) >> 8;

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);
}

GPixel src_atop_mode(GPixel src, GPixel dst) {
       uint8_t srcA = GPixel_GetA(src);
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);

    uint8_t dstA = GPixel_GetA(dst);
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);

    float dstAlpha = dstA / 255.0f;
    float srcAlphaComp = 1.0f - (srcA / 255.0f);

    uint8_t resultA = (int)(std::min(255.0f, (srcA * dstAlpha) + (dstA * srcAlphaComp) + 0.5f));
    uint8_t resultR = (int)((srcR * dstAlpha) + (dstR * srcAlphaComp) + 0.5f);
    uint8_t resultG = (int)((srcG * dstAlpha) + (dstG * srcAlphaComp) + 0.5f);
    uint8_t resultB = (int)((srcB * dstAlpha) + (dstB * srcAlphaComp) + 0.5f);

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);

}

GPixel dst_atop_mode(GPixel src, GPixel dst) {
    uint8_t srcA = GPixel_GetA(src);
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);

    uint8_t dstA = GPixel_GetA(dst);
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);

    float srcAlpha = srcA / 255.0f;
    float dstAlphaComp = 1.0f - (dstA / 255.0f);

    uint8_t resultA = (int)(std::min(255.0f, (dstA * srcAlpha) + (srcA * dstAlphaComp) + 0.5f));
    uint8_t resultR = (int)((dstR * srcAlpha) + (srcR * dstAlphaComp) + 0.5f);
    uint8_t resultG = (int)((dstG * srcAlpha) + (srcG * dstAlphaComp) + 0.5f);
    uint8_t resultB = (int)((dstB * srcAlpha) + (srcB * dstAlphaComp) + 0.5f);

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);
}

GPixel xor_mode(GPixel src, GPixel dst) {
    uint8_t srcA = GPixel_GetA(src);
    uint8_t srcR = GPixel_GetR(src);
    uint8_t srcG = GPixel_GetG(src);
    uint8_t srcB = GPixel_GetB(src);

    uint8_t dstA = GPixel_GetA(dst);
    uint8_t dstR = GPixel_GetR(dst);
    uint8_t dstG = GPixel_GetG(dst);
    uint8_t dstB = GPixel_GetB(dst);

    float srcAlphaComp = 1.0f - (srcA / 255.0f);
    float dstAlphaComp = 1.0f - (dstA / 255.0f);

    uint8_t resultA = (int)(srcA * dstAlphaComp + dstA * srcAlphaComp + 0.5f);
    uint8_t resultR = (int)(srcR * dstAlphaComp + dstR * srcAlphaComp + 0.5f);
    uint8_t resultG = (int)(srcG * dstAlphaComp + dstG * srcAlphaComp + 0.5f);
    uint8_t resultB = (int)(srcB * dstAlphaComp + dstB * srcAlphaComp + 0.5f);

    return GPixel_PackARGB(resultA, resultR, resultG, resultB);
}
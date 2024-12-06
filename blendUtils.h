#ifndef BLEND_UTILS_H
#define BLEND_UTILS_H

#include "include/GPixel.h"
#include "include/GBlendMode.h"

GPixel clear_mode(GPixel src, GPixel dst);
GPixel src_mode(GPixel src, GPixel dst);
GPixel dst_mode(GPixel src, GPixel dst);
GPixel src_over_mode(GPixel src, GPixel dst);
GPixel dst_over_mode(GPixel src, GPixel dst);
GPixel src_in_mode(GPixel src, GPixel dst);
GPixel dst_in_mode(GPixel src, GPixel dst);
GPixel src_out_mode(GPixel src, GPixel dst);
GPixel dst_out_mode(GPixel src, GPixel dst);
GPixel src_atop_mode(GPixel src, GPixel dst);
GPixel dst_atop_mode(GPixel src, GPixel dst);
GPixel xor_mode(GPixel src, GPixel dst);

using BlendProc = GPixel (*)(GPixel src, GPixel dst);
BlendProc findBlend(GBlendMode mode);

#endif // BLEND_UTILS_H
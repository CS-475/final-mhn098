#include "MyGFinal.h"
#include <cmath>
#include <vector>

std::unique_ptr<GFinal> GCreateFinal() {
    return std::make_unique<MyGFinal>();
}

namespace {
    std::shared_ptr<GPath> final_addStrokedLine(GPoint p0, GPoint p1, float width, bool roundCap) {
        float slope = (p1.y - p0.y) / (p1.x - p0.x);
        float perp = -1 / slope;

        float corner_angle = atan(perp);

        float corner_dx = cos(corner_angle) * width / 2;
        float corner_dy = sin(corner_angle) * width / 2;

        GPoint c0 = {p0.x + corner_dx, p0.y + corner_dy};
        GPoint c1 = {p0.x - corner_dx, p0.y - corner_dy};
        GPoint c2 = {p1.x - corner_dx, p1.y - corner_dy};
        GPoint c3 = {p1.x + corner_dx, p1.y + corner_dy};

        GPathBuilder builder;
        builder.moveTo(c0);
        builder.lineTo(c1);
        builder.lineTo(c2);
        builder.lineTo(c3);

        return builder.detach();
    }
}

std::shared_ptr<GShader> MyGFinal::createVoronoiShader(const GPoint points[], const GColor colors[], int count) {
    return nullptr;
}

std::shared_ptr<GPath> MyGFinal::strokePolygon(const GPoint points[], int count, float width, bool isClosed) {
    if (count < 2) return nullptr;

    GPathBuilder builder;
    float halfWidth = width / 2;

    auto getNormal = [](GPoint p0, GPoint p1, float scale) -> std::pair<float, float> {
        float dx = p1.x - p0.x;
        float dy = p1.y - p0.y;
        float len = sqrt(dx * dx + dy * dy);
        if (len == 0) return {0, 0};
        return {(-dy / len) * scale, (dx / len) * scale};
    };

    auto [nx, ny] = getNormal(points[0], points[1], halfWidth);
    builder.moveTo({points[0].x + nx, points[0].y + ny});

    for (int i = 0; i < count - 1; i++) {
        GPoint curr = points[i];
        GPoint next = points[i + 1];
        auto [nx1, ny1] = getNormal(curr, next, halfWidth);
        builder.lineTo({next.x + nx1, next.y + ny1});
    }

    for (int i = count - 1; i > 0; i--) {
        GPoint curr = points[i];
        GPoint prev = points[i - 1];
        auto [nx, ny] = getNormal(prev, curr, halfWidth);
        builder.lineTo({curr.x - nx, curr.y - ny});
    }

    if (isClosed) {
        GPoint first = points[0];
        GPoint last = points[count - 1];
        auto [nx, ny] = getNormal(last, first, halfWidth);
        builder.lineTo({first.x + nx, first.y + ny});
    } else {
        auto [nx, ny] = getNormal(points[0], points[1], halfWidth);
        builder.lineTo({points[0].x + nx, points[0].y + ny});
    }

    return builder.detach();
}

std::shared_ptr<GShader> MyGFinal::createLinearPosGradient(GPoint p0, GPoint p1, const GColor colors[], const float pos[], int count) {
    if (count < 1 || !colors || !pos) {
        return nullptr;
    }

    std::vector<GColor> interpolatedColors;
    for (float t = 0; t <= 1.0f; t += 0.01f) {
        int index = 0;
        while (index < count - 1 && t > pos[index + 1]) {
            index++;
        }

        GColor color;
        if (t <= pos[0]) {
            color = colors[0];
        } else if (t >= pos[count - 1]) {
            color = colors[count - 1];
        } else {
            float t0 = pos[index];
            float t1 = pos[index + 1];
            float weight = (t - t0) / (t1 - t0);

            const GColor& c0 = colors[index];
            const GColor& c1 = colors[index + 1];
            color = {c0.r + (c1.r - c0.r) * weight, c0.g + (c1.g - c0.g) * weight, c0.b + (c1.b - c0.b) * weight, c0.a + (c1.a - c0.a) * weight};
        }

        interpolatedColors.push_back(color);
    }
    return GCreateLinearGradient(p0, p1, interpolatedColors.data(), interpolatedColors.size(), GTileMode::kClamp);
}
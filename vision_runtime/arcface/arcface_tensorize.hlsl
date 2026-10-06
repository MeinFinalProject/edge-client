Texture2D<float>  InputY  : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp  : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);

cbuffer Params : register(b0)
{
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;

    float Inv00;
    float Inv01;
    float Inv02;
    float _Pad0;

    float Inv10;
    float Inv11;
    float Inv12;
    float _Pad1;
};

// Same validated camera color path used by the native SCRFD/PAD runtime.
// The current 1280x720 camera surface is treated as limited-range BT.709 NV12.
float3 Nv12Bt709LimitedToRgb(float ySample, float2 uvSample)
{
    float y  = (ySample * 255.0f - 16.0f) / 219.0f;
    float cb = (uvSample.x * 255.0f - 128.0f) / 224.0f;
    float cr = (uvSample.y * 255.0f - 128.0f) / 224.0f;

    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.187324f * cb - 0.468124f * cr;
    rgb.b = y + 1.8556f * cb;
    return saturate(rgb);
}

float3 SampleRgbPointBorder(int sx, int sy)
{
    if (sx < 0 || sy < 0 || sx >= (int)SrcWidth || sy >= (int)SrcHeight)
        return float3(0.0f, 0.0f, 0.0f);

    const float y = InputY.Load(int3(sx, sy, 0));
    const float2 uvCoord = float2(
        ((float)sx + 0.5f) / (float)SrcWidth,
        ((float)sy + 0.5f) / (float)SrcHeight
    );
    const float2 uv = InputUV.SampleLevel(LinearClamp, uvCoord, 0.0f);
    return Nv12Bt709LimitedToRgb(y, uv);
}

// cv::warpAffine uses INTER_LINEAR by default in InsightFace face_align.norm_crop.
// Perform bilinear interpolation after applying the inverse similarity map.
float3 SampleRgbLinear(float sx, float sy)
{
    const int x0 = (int)floor(sx);
    const int y0 = (int)floor(sy);
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;

    const float3 p00 = SampleRgbPointBorder(x0,     y0);
    const float3 p10 = SampleRgbPointBorder(x0 + 1, y0);
    const float3 p01 = SampleRgbPointBorder(x0,     y0 + 1);
    const float3 p11 = SampleRgbPointBorder(x0 + 1, y0 + 1);

    const float3 top = lerp(p00, p10, fx);
    const float3 bot = lerp(p01, p11, fx);
    return lerp(top, bot, fy);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= DstWidth || tid.y >= DstHeight)
        return;

    // OpenCV warpAffine evaluates the inverse mapping at integer destination
    // pixel coordinates. Inv maps aligned ArcFace output -> camera source.
    const float u = (float)tid.x;
    const float v = (float)tid.y;
    const float sx = Inv00 * u + Inv01 * v + Inv02;
    const float sy = Inv10 * u + Inv11 * v + Inv12;

    const float3 rgb01 = SampleRgbLinear(sx, sy);
    const float3 pixel255 = rgb01 * 255.0f;

    // InsightFace w600k_r50 preprocessing for the buffalo_l recognition model.
    const float3 normalized = (pixel255 - 127.5f) / 127.5f;

    const uint plane = DstWidth * DstHeight;
    const uint pixel = tid.y * DstWidth + tid.x;

    // NCHW RGB. Official Python uses cv2.dnn.blobFromImages(... swapRB=True).
    OutputTensor.Store((pixel) * 4, asuint(normalized.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(normalized.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(normalized.b));
}
Texture2D<float>  InputY  : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp  : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);
RWByteAddressBuffer Coverage : register(u1); // 25 group sums, then one float ratio
groupshared uint ValidWeights[256];

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
    // BORDER_REPLICATE: clamp each tap before reading, not the crop box.
    sx = clamp(sx, 0, (int)SrcWidth - 1);
    sy = clamp(sy, 0, (int)SrcHeight - 1);

    const float y = InputY.Load(int3(sx, sy, 0));
    const float2 uvCoord = float2(
        ((float)sx + 0.5f) / (float)SrcWidth,
        ((float)sy + 0.5f) / (float)SrcHeight
    );
    const float2 uv = InputUV.SampleLevel(LinearClamp, uvCoord, 0.0f);
    // SDK warps a uint8 RGB/BGR image, then converts the crop to float32.
    return floor(Nv12Bt709LimitedToRgb(y, uv) * 255.0f + 0.5f);
}

uint ValidTap(int x, int y)
{
    return x >= 0 && y >= 0 && x < (int)SrcWidth && y < (int)SrcHeight ? 1 : 0;
}

float3 SampleRgbLinear(int x0, int y0, float fx, float fy)
{
    const float3 p00 = SampleRgbPointBorder(x0,     y0);
    const float3 p10 = SampleRgbPointBorder(x0 + 1, y0);
    const float3 p01 = SampleRgbPointBorder(x0,     y0 + 1);
    const float3 p11 = SampleRgbPointBorder(x0 + 1, y0 + 1);

    const float3 top = lerp(p00, p10, fx);
    const float3 bot = lerp(p01, p11, fx);
    return floor(lerp(top, bot, fy) + 0.5f) / 255.0f;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    // The validated 80x80 contract dispatches exactly 5x5 groups of 16x16.
    // No early return may bypass a group barrier.
    const float u = (float)tid.x;
    const float v = (float)tid.y;
    // OpenCV INTER_LINEAR affine coordinate table: AB_BITS=10, INTER_BITS=5.
    // Separate rounding of the x delta and y/translation term matters at borders.
    const int X = ((int)round(Inv00*u*1024.0f) + (int)round((Inv01*v+Inv02)*1024.0f) + 16) >> 5;
    const int Y = ((int)round(Inv10*u*1024.0f) + (int)round((Inv11*v+Inv12)*1024.0f) + 16) >> 5;
    const int x0 = X >> 5;
    const int y0 = Y >> 5;
    const uint fx = X & 31;
    const uint fy = Y & 31;
    // Warp an implicit all-ones mask with zero border. Integer weights sum to
    // 1024 per fully covered pixel; retain fractional coverage at the edge.
    ValidWeights[lane] =
        ValidTap(x0,y0)*(32-fx)*(32-fy) + ValidTap(x0+1,y0)*fx*(32-fy) +
        ValidTap(x0,y0+1)*(32-fx)*fy + ValidTap(x0+1,y0+1)*fx*fy;
    const float3 normalized = SampleRgbLinear(x0, y0, fx/32.0f, fy/32.0f);

    const uint plane = DstWidth * DstHeight;
    const uint pixel = tid.y * DstWidth + tid.x;

    // NCHW RGB.
    OutputTensor.Store((pixel) * 4, asuint(normalized.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(normalized.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(normalized.b));
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) ValidWeights[lane] += ValidWeights[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) Coverage.Store((group.y*5+group.x)*4, ValidWeights[0]);
}

[numthreads(32, 1, 1)]
void reduce(uint lane : SV_GroupIndex)
{
    ValidWeights[lane] = lane < 25 ? Coverage.Load(lane*4) : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) ValidWeights[lane] += ValidWeights[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) Coverage.Store(100, asuint(saturate(1.0f - ValidWeights[0]/6553600.0f)));
}
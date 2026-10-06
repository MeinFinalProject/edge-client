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

    float Scale;
    float PadX;
    float PadY;
    float Reserved;
};

float3 Nv12Bt709LimitedToRgb(float ySample, float2 uvSample)
{
    // UVC 720p NV12 umumnya memakai studio/limited range dan matriks BT.709.
    // Nilai dikembalikan ke domain nominal 8-bit sebelum matrix YUV->RGB.
    float y  = (ySample  * 255.0f - 16.0f)  / 219.0f;
    float cb = (uvSample.x * 255.0f - 128.0f) / 224.0f;
    float cr = (uvSample.y * 255.0f - 128.0f) / 224.0f;

    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.187324f * cb - 0.468124f * cr;
    rgb.b = y + 1.8556f * cb;

    return saturate(rgb);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= DstWidth || tid.y >= DstHeight)
        return;

    const uint plane = DstWidth * DstHeight;
    const uint pixelIndex = tid.y * DstWidth + tid.x;

    // InsightFace/SCRFD membuat letterbox hitam (pixel 0), lalu:
    // blob = (pixel - 127.5) / 128.
    const float normalizedBlack = -127.5f / 128.0f;

    float3 normalized = float3(
        normalizedBlack,
        normalizedBlack,
        normalizedBlack
    );

    const float x = (float)tid.x + 0.5f;
    const float y = (float)tid.y + 0.5f;

    const float contentWidth  = (float)SrcWidth  * Scale;
    const float contentHeight = (float)SrcHeight * Scale;

    if (
        x >= PadX && x < (PadX + contentWidth) &&
        y >= PadY && y < (PadY + contentHeight)
    ) {
        const float srcX = (x - PadX) / Scale;
        const float srcY = (y - PadY) / Scale;

        const float2 uv = float2(
            srcX / (float)SrcWidth,
            srcY / (float)SrcHeight
        );

        const float ySample = InputY.SampleLevel(LinearClamp, uv, 0.0f);
        const float2 uvSample = InputUV.SampleLevel(LinearClamp, uv, 0.0f);

        const float3 rgb = Nv12Bt709LimitedToRgb(ySample, uvSample);
        normalized = (rgb * 255.0f - 127.5f) / 128.0f;
    }

    // Original SCRFD input contract: NCHW RGB float32.
    // RWByteAddressBuffer dipakai agar D3D11On12 dapat menulis langsung ke
    // backing buffer D3D12 yang juga dipakai TensorFloat WinML.
    OutputTensor.Store((pixelIndex) * 4, asuint(normalized.r));
    OutputTensor.Store((plane + pixelIndex) * 4, asuint(normalized.g));
    OutputTensor.Store((2 * plane + pixelIndex) * 4, asuint(normalized.b));
}
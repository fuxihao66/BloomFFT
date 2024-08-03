
cbuffer cb0 : register(b0)
{
    uint4 DstRect;
    float3 BrightPixelGain;
    uint Width;
    uint Height;
    uint TransformType;
    uint InputTextureOffset;
    uint OutputTextureOffset;
    uint DstPostFilterParaOffset;
};

// 需要用preview版本的dxc
// 不能链接dxil.dll 还是说要用别的版本的dxil
// 用最新的release dxil.dll 会报错误？

[numthreads(16, 16, 1)]
void CSMain(
    uint2 dispatchThreadId : SV_DispatchThreadID,
    uint2 groupTheadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID,
    uint groupIndex : SV_GroupIndex
    )
{
    const bool bIsHorizontal = (TransformType & 0x1);
	const bool bIsForward    = (TransformType & 0x2);
	const bool bModifyInput  = 1;


    Texture2D<float4> SrcTexture = ResourceDescriptorHeap[InputTextureOffset];
    RWTexture2D<float4> OutputTexture = ResourceDescriptorHeap[OutputTextureOffset];

    OutputTexture[dispatchThreadId] = SrcTexture[dispatchThreadId];
}
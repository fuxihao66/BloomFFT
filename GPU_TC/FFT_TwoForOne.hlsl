// #define SCAN_LINE_LENGTH
#define WAVE_MATRIX_DIM 16
#define WAVE_SIZE 32
#ifndef SCAN_LINE_LENGTH
    #define SCAN_LINE_LENGTH 2048
#endif

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
	uint FBufferOffset;
	uint FBufferInverseOffset;
};
#include "FFT_Common.hlsl"

// 需要用preview版本的dxc
// 不能链接dxil.dll 还是说要用别的版本的dxil
// 用最新的release dxil.dll 会报错误？

// #define TILE_SIZE 16

// // Shared Memory
// groupshared float16_t groupMatInput_x[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatInput_y[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatInput_z[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatWeight[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatOutput_x[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatOutput_y[TILE_SIZE * TILE_SIZE];
// groupshared float16_t groupMatOutput_z[TILE_SIZE * TILE_SIZE];

float ConvertToLuma(in float3 ColorValue)
{
	// Rec 709 function for luma.
	return dot(ColorValue, float3(0.2126, 0.7152, 0.0722));
}
bool FilterPixel(in float3 Filter, inout float4 PixelValue)
{
	bool bIsChanged = false;
	float Luma = ConvertToLuma(PixelValue.xyz);
	
	if (Luma > Filter.x)
	{
		float TargetLuma = Filter.z * (Luma - Filter.x) + Filter.x;
		TargetLuma  = min(TargetLuma, Filter.y);

		PixelValue.rgb *= (TargetLuma / Luma);
		bIsChanged = true;
	}

	return bIsChanged;
}
void ModifyInput(inout Complex LocalBuffer[2][RADIX])
{
	for (uint r = 0; r < RADIX; ++r)
	{
		float4 InputColor = float4(LocalBuffer[0][r].x, LocalBuffer[0][r].y, LocalBuffer[1][r].x, LocalBuffer[1][r].y);

		bool bIsChanged = FilterPixel(BrightPixelGain, InputColor);
		
		if (bIsChanged)
		{
			LocalBuffer[0][r] = InputColor.xy;
			LocalBuffer[1][r] = InputColor.zw;
		}
	}
}
#define STRIDE ( SCAN_LINE_LENGTH / RADIX )

// every thread deals with 16
[wavesize(WAVE_SIZE)]
[numthreads(SCAN_LINE_LENGTH / 16, 1, 1)]
void CSMain(
    uint2 groupTheadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID,
    uint groupIndex : SV_GroupIndex
    )
{
    const bool bIsHorizontal = (TransformType & 0x1);
	const bool bIsForward    = (TransformType & 0x2);
	const bool bModifyInput  = (TransformType & 0x4);

	const uint SignalLength = SCAN_LINE_LENGTH;
	const uint ThreadIdx = groupTheadId.x;
	const uint ScanIdx  = groupId.z;
    uint Head = ThreadIdx;
	const uint Stride = STRIDE;

	Complex LocalBuffer[2][RADIX];

    if (bIsForward) 
	{
		uint4 SrcRect;
		SrcRect.xy = uint2(0, 0);
		SrcRect.zw = uint2(Width, Height);
		// Read from an image buffer
		
		CopyDataSrcWindowToLocal(LocalBuffer, bIsHorizontal, ScanIdx, Head, Stride, SrcRect);

		
		ScrubNANs(LocalBuffer);

		if (bModifyInput)
		{
			// The input colors may be modified to boost the bright pixels.
			ModifyInput(LocalBuffer);	
		}		
	}
    else 
	{
		// Read a frequency space buffer with two-for-one data layout
		ReadTwoForOneFrequencyData(bIsHorizontal, LocalBuffer, ScanIdx, Head, Stride, SignalLength);
	}
	GroupSharedTCFFT(bIsForward, LocalBuffer, SignalLength, ThreadIdx);

    if (bIsForward) 
	{
		WriteTwoForOneFrequencyData(bIsHorizontal, LocalBuffer, ScanIdx, Head, Stride, SignalLength);
	}
	else
	{
		ScrubNANs(LocalBuffer);
		
        StructuredBuffer<float4> DstPostFilterParameters = ResourceDescriptorHeap[DstPostFilterParaOffset];
		float4 Scale = DstPostFilterParameters[0];

		[unroll]
		for (uint r = 0; r < RADIX; r++)
		{
			LocalBuffer[0][r] *= Scale.xy;
			LocalBuffer[1][r] *= Scale.xw;
		}

		CopyDataLocalToDstWindow(LocalBuffer, bIsHorizontal, ScanIdx, Head, Stride, DstRect);
	}
    
}
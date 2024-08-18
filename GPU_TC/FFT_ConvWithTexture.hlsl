#define WAVE_MATRIX_DIM 16
#define WAVE_SIZE 32
#ifndef SCAN_LINE_LENGTH
    #define SCAN_LINE_LENGTH 2048
#endif
cbuffer cb0 : register(b0)
{
    uint2 SrcRectMax;
    uint2 DstExtent;
    uint TransformType;
    uint InputTextureOffset;
    uint FilterTextureOffset;
    uint OutputTextureOffset;
	uint FBufferOffset;
	uint FBufferInverseOffset;
};
#include "FFT_Common.hlsl"

#define NUMTHREADSX ( SCAN_LINE_LENGTH / RADIX )
#define STRIDE ( SCAN_LINE_LENGTH / RADIX )


void CopyFilterTextureToFilterBuffer(inout Complex Filter[2][RADIX], bool bIsHorizontal, in uint N, in uint Head, in uint Stride, in uint ScanIdx)
{
    Texture2D FilterTexture = ResourceDescriptorHeap[FilterTextureOffset];
	if (bIsHorizontal) 
	{
		uint2 Pixel = uint2(Head, ScanIdx);
		[unroll]
		for (uint r = 0; r < RADIX; ++r, Pixel.x += Stride)
		{	
			float4 TextureValue = FilterTexture[Pixel];
			Filter[0][r] = TextureValue.xy;
			Filter[1][r] = TextureValue.zw;
		}
	
	}
	else 
	{
		uint2 Pixel = uint2(ScanIdx, Head);
		[unroll]
		for (uint r = 0; r < RADIX; ++r, Pixel.y += Stride)
		{
			float4 TextureValue = FilterTexture[Pixel];
			Filter[0][r] = TextureValue.xy;
			Filter[1][r] = TextureValue.zw;
		}
	}

}
void ComplexMultTexture( bool bUseAlpha, bool bIsGAGroup,  in Complex Filter[2][RADIX], inout Complex LocalBuffer[2][RADIX])
{
	[unroll]
    for (uint r = 0; r < RADIX; ++r)
	{
		LocalBuffer[0][r] = ComplexMult(LocalBuffer[0][r], Filter[0][r]);
	}   
	if (bUseAlpha || !bIsGAGroup)
	{
		[unroll] 
        for (uint r = 0; r < RADIX; ++r)
		{
			LocalBuffer[1][r] = ComplexMult(LocalBuffer[1][r], Filter[1][r]);
		}
	}
}
void ComplexMultTexture(bool bIsHorizontal, bool bUseAlpha, bool bIsGAGroup, in uint N, in uint Head, in uint Stride, in uint ScanIdx, inout Complex LocalBuffer[2][RADIX])
{
	Complex Filter[2][RADIX];
	CopyFilterTextureToFilterBuffer(Filter, bIsHorizontal, N, Head, Stride, ScanIdx);
	
	ComplexMultTexture( bUseAlpha, bIsGAGroup,  Filter, LocalBuffer);
}
void GetKernelSum(in Texture2D KernelTexture, in bool bIsHorizontal, uint NumScanlines, inout float2 Integral[2])
{

	
	if (!bIsHorizontal)  // Since this is the second tranform the first must have been horizontal, assume the data layout from a 2-for-1
	{
		Integral[0] = KernelTexture[uint2(0, 0)].xz;  // RB sums
		Integral[1] = KernelTexture[uint2(NumScanlines-2, 0)].xz;
		
	}	
	else
	{
		Integral[0] = KernelTexture[uint2(0, 0)].xz;
		Integral[1] = KernelTexture[uint2(0, NumScanlines-2)].xz;
	}
}

[wavesize(WAVE_SIZE)]
[numthreads(NUMTHREADSX, 1, 1)]
void GSConvolutionWithTextureCS(uint3 GroupID : SV_GroupID, uint3 GroupThreadID : SV_GroupThreadID )
{
    Texture2D FilterTexture = ResourceDescriptorHeap[FilterTextureOffset];

	// These are constant across all thread groups
	const bool bIsHorizontal = (TransformType & 0x1);
	const bool bIsForward    = (TransformType & 0x2);
	const bool bUseAlpha     = (TransformType & 0x8);

	// Threads are defined in a 1d array.

	const uint ThreadIdx = GroupThreadID.x;

	// The scan line this thread works on

	const uint ScanIdx  = GroupID.z;
	 
	const uint NumScanlines = (bIsHorizontal) ? DstExtent.y : DstExtent.x;

	//const uint NumFrequencies = TransformSize.x * TransformSize.y;

	// The two-for-one transform results in a data layout with complex coefficients
	// R G B A (representing the 1-d transform of r g b a)
	// In half of the buffer 
	//       R = SrcTexture.xy, B = SrcTexture.zw 
	// and in the other half
	//       G = SrcTexture.xy, A = SrcTexture.zw
	// With this layout 
	// R/B = columns [0, .., NumScanlines/2 -1] 
	// G/A = columns [NumScanlines/2, .., NumScanlines-1]

	// This data is loaded into the LocalBuffer[2][RADIX]
	// as   
	// LocalBuffer[0] = {R | G};  
	// LocalBuffer[1] = {B | A};
	
	
	// The thread groups in this shader act on the columns. 


	// This thread group acts on the columns of G/A.
	const bool bIsGAGroup = (2 * ScanIdx > NumScanlines - 2 );
	
	// Force the alpha 'tint' to do nothing
	float4 FilterTint = float4(1, 1, 1, 1);
	const Complex Tint = (bIsGAGroup) ? Complex(FilterTint.y, 1.f) : FilterTint.xz;

	// The length of the signal to be transformed
	
	const uint SignalLength = SCAN_LINE_LENGTH;

	// The main memory access pattern for this thread.

	uint Head = ThreadIdx;
	const uint Stride = STRIDE;   
	 
	// Thread-local memory.  Reserve two arrays since we plit .xy and .zw channels   
	
	Complex LocalBuffer[2][RADIX];
	
	// Load the filter
	Complex Filter[2][RADIX];
	CopyFilterTextureToFilterBuffer(Filter, bIsHorizontal, SignalLength, Head, Stride, ScanIdx);
	
	// Load the local memory from the source texture
	// LocalBuffer[0][] holds .xy,  LocalBuffer[1][] holds.zw
    const uint2 SrcRectMin = {0, 0}; 
	CopyDataSrcWindowToLocal(LocalBuffer, bIsHorizontal, ScanIdx, Head, Stride, SrcRectMin, SrcRectMax);


	// Fourier Transform  the data
	// This uses the group shared memory and has appropriate syncs
	
	GroupSharedTCFFT(bIsForward, LocalBuffer, SignalLength, Head);
	
	 
	// ---- Convolution in frequency space is a multiply.
	// Here we multiply against the transform of a physical space kernel, but special case the 
	// thread groups that are working on  Green and Alpha

	{
		float2 Norm[2];
		GetKernelSum(FilterTexture, bIsHorizontal, NumScanlines, Norm);
		// redSum = Norm[0].x;  greenSum = Norm[0].y; blueSum = Norm[1].x; alphaSum = Norm[1].y
		
		// Normalize R & G
		{
			//float Normal = NormMax;
			float Normal = (bIsGAGroup) ? Norm[1].x :  Norm[0].x;

			for (uint r = 0; r < RADIX; ++r)
			{
				// This is the R or G channel
				LocalBuffer[0][r] /= Normal; 
			}
		}
		
		// Normalize B & A
		{
			//float Normal = AorBNorm;
			float Normal = (bIsGAGroup) ? Norm[1].y :  Norm[0].y;

			for (uint r = 0; r < RADIX; ++r)
			{
				// This is the B or A channel
				LocalBuffer[1][r] /= Normal; 
			}
		}
	}


	ComplexMultTexture( bUseAlpha, bIsGAGroup,  Filter, LocalBuffer);

	// The input kernel might not have been normalized.  
	// This applies the correct normalization to local buffer,

	

	// ---- Transform back ---- //

	GroupSharedTCFFT(!bIsForward, LocalBuffer, SignalLength, Head);

	// Apply additional tinting to the convolution result

	// ApplyTint(Tint, LocalBuffer);// NOT USED

	// Copy Data back to main memory (dst)
	//uint2 Extent = SrcRectMax - SrcRectMin;

	CopyDataLocalToDstWindow(LocalBuffer, bIsHorizontal, ScanIdx, Head, Stride, DstExtent);
}
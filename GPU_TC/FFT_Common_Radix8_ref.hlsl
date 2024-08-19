#define RADIX 8
#define Complex float16_t2
static const float PI = 3.14159265f;

void ScrubNANs(inout Complex LocalBuffer[2][RADIX])
{

	[unroll]
	for (uint r = 0; r < RADIX; ++r)
	{	
	
		LocalBuffer[0][r] = -min(-LocalBuffer[0][r], Complex(0,0) ); 
		LocalBuffer[1][r] = -min(-LocalBuffer[1][r], Complex(0,0) ); 
	}
}


void CopyDataSrcWindowToLocal(inout Complex LocalBuffer[2][RADIX], bool bIsHorizontal, uint ScanIdx, uint Loc, uint Stride, uint2 WindowMin, uint2 WindowMax )
{
	{ for (uint i = 0; i < RADIX; ++i) LocalBuffer[0][ i ] = Complex(0.f, 0.f); }
	{ for (uint i = 0; i < RADIX; ++i) LocalBuffer[1][ i ] = Complex(0.f, 0.f); }
    Texture2D SrcTexture = ResourceDescriptorHeap[InputTextureOffset];

	if (bIsHorizontal) 
	{
		uint2 Pixel = uint2(Loc, ScanIdx) + uint2(WindowMin.x, 0);
		[unroll]
		for (uint i = 0; i < RADIX; ++i, Pixel.x += Stride)
		{	
			bool InWindow = Pixel.x < WindowMax.x; 
			if (InWindow)
			{ 
				float4 SrcValue = SrcTexture[Pixel];
				LocalBuffer[0][ i ] = SrcValue.xy;
				LocalBuffer[1][ i ] = SrcValue.zw;
			}	
			else
			{
				LocalBuffer[0][ i ] = 0.0;
				LocalBuffer[1][ i ] = 0.0;
			}	
		}
	}
	else 
	{
		uint2 Pixel = uint2(ScanIdx, Loc) + uint2(0, WindowMin.y);
		[unroll]
		for (uint i = 0; i < RADIX; ++i, Pixel.y += Stride)
		{
			bool InWindow = Pixel.y < WindowMax.y;
			if (InWindow)
			{	
				float4 SrcValue = SrcTexture[Pixel];
				LocalBuffer[0][ i ] = SrcValue.xy;
				LocalBuffer[1][ i ] = SrcValue.zw;
			}
			else
			{
				LocalBuffer[0][ i ] = 0.0;
				LocalBuffer[1][ i ] = 0.0;
			}
		}
	}
}
void CopyDataSrcWindowToLocal(inout Complex LocalBuffer[2][RADIX], bool bIsHorizontal, in uint ScanIdx, uint Loc, uint Stride, uint4 Window)
{
	{ for (uint i = 0; i < RADIX; ++i) LocalBuffer[0][ i ] = Complex(0.f, 0.f); }
	{ for (uint i = 0; i < RADIX; ++i) LocalBuffer[1][ i ] = Complex(0.f, 0.f); }
    Texture2D SrcTexture = ResourceDescriptorHeap[InputTextureOffset];
	
	if (bIsHorizontal) 
	{
		// offset for window start
		uint2 Pixel = uint2(Loc, ScanIdx) + Window.xy;
		[unroll]
		for (uint i = 0; i < RADIX ; ++i, Pixel.x += Stride)
		{	
			bool InWindow = Pixel.x < Window.z; 
			if (InWindow)
			{ 
				float4 SrcValue = SrcTexture[Pixel];
				LocalBuffer[0][ i ] = SrcValue.xy;
				LocalBuffer[1][ i ] = SrcValue.zw;
			}
			else
			{
				LocalBuffer[0][ i ] = 0.0;
				LocalBuffer[1][ i ] = 0.0;
			}

		}
	}
	else 
	{
		// offset for window start
		uint2 Pixel = uint2(ScanIdx, Loc) + Window.xy;
		[unroll]
		for (uint i = 0; i < RADIX; ++i, Pixel.y += Stride)
		{
			bool InWindow = Pixel.y < Window.w;
			if (InWindow)
			{ 
				float4 SrcValue = SrcTexture[Pixel];
				LocalBuffer[0][ i ] = SrcValue.xy;
				LocalBuffer[1][ i ] = SrcValue.zw;
			}
			else
			{
				LocalBuffer[0][ i ] = 0.0;
				LocalBuffer[1][ i ] = 0.0;
			}
		}
	}
}
#ifndef NORMALIZE_ORDER
#define NORMALIZE_ORDER 1
#endif
float ForwardScale(uint ArrayLength)
{	
#if NORMALIZE_ORDER  == 1
	return float(1); 
#else
	return (float(32) / float(ArrayLength) );
#endif
}

float InverseScale(uint ArrayLength)
{
#if NORMALIZE_ORDER == 1
	return ( float(1) / float(ArrayLength) );
#else
	return float(1) / 32.f;
#endif
}

void Scale(inout Complex LocalBuffer[2][RADIX], in float ScaleValue)
{
	// Scale
	{
		for (uint r = 0; r < RADIX; ++r)
		{	
			LocalBuffer[0][r] *= ScaleValue; 
		}
	}

	{
		for (uint r = 0; r < RADIX; ++r)
		{	
			LocalBuffer[1][r] *= ScaleValue; 
		}
	}
}
Complex ComplexMult(in Complex A, in Complex B)
{
	return Complex(A.x * B.x - A.y * B.y, A.x * B.y + B.x * A.y);
}
void Radix2FFT(in bool bIsForward, inout Complex V0, inout Complex V1)
{
	V0 = V0 + V1;
	V1 = V0 - V1 - V1;
}
void Radix4FFT(in bool bIsForward, inout Complex V0, inout Complex V1, inout Complex V2, inout Complex V3)
{
	// The even and odd transforms
	Radix2FFT(bIsForward, V0, V2); 
	Radix2FFT(bIsForward, V1, V3); 

	// The butterfly merge of the even and odd transforms
	Complex Tmp;
	Complex TmpV1 = V1;
	 
	if (bIsForward) {
		// Complex(0, 1) * V3
		Tmp = Complex(V3.y, -V3.x);
	}
	else 
	{
		// Complex(0, -1) * V3
		Tmp = Complex(-V3.y, V3.x);
	}

	V0 = V0 + TmpV1;
	V1 = V2 + Tmp;
	V3 = V2 - Tmp;
	V2 = V0 - TmpV1 - TmpV1;
}
void Radix8FFT(in bool bIsForward, inout Complex V0, inout Complex V1, inout Complex V2, inout Complex V3, inout Complex V4, inout Complex V5, inout Complex V6, inout Complex V7)
{
	// The even and odd transforms
	Radix4FFT(bIsForward, V0, V2, V4, V6);
	Radix4FFT(bIsForward, V1, V3, V5, V7);

	Complex Twiddle;
	// 0.7071067811865475 = 1/sqrt(2)
	float InvSqrtTwo = float(1.f) / sqrt(2.f);
	if (bIsForward)
	{
		Twiddle = Complex(InvSqrtTwo, -InvSqrtTwo);
	}
	else
	{
		Twiddle = Complex(InvSqrtTwo, InvSqrtTwo);
	}

	Complex Rslt[8];
	Complex Tmp = ComplexMult(Twiddle, V3);

	Rslt[0] = V0 + V1;
	Rslt[4] = V0 - V1;

	Rslt[1] = V2 + Tmp;
	Rslt[5] = V2 - Tmp;

	if (bIsForward)
	{
		// V4 + i V5
		Rslt[2] = Complex(V4.x + V5.y, V4.y - V5.x);
		// V4 - i V5
		Rslt[6] = Complex(V4.x - V5.y, V4.y + V5.x);
	}
	else
	{
		// V4 - iV5
		Rslt[2] = Complex(V4.x - V5.y, V4.y + V5.x);
		// V4 + iV5
		Rslt[6] = Complex(V4.x + V5.y, V4.y - V5.x);
	}	

	Twiddle.x = -Twiddle.x;
	Tmp = ComplexMult(Twiddle, V7);

	Rslt[3] = V6 + Tmp;
	Rslt[7] = V6 - Tmp;

	V0 = Rslt[0];
	V1 = Rslt[1];
	V2 = Rslt[2];
	V3 = Rslt[3];
	V4 = Rslt[4];
	V5 = Rslt[5];
	V6 = Rslt[6];
	V7 = Rslt[7];
}
void Radix8FFT(in bool bIsForward, inout Complex Local[8])
{
    Radix8FFT(bIsForward, Local[0], Local[1], Local[2], Local[3], Local[4], Local[5], Local[6], Local[7]);
}

void RadixFFT(in bool bIsForward, inout Complex v[RADIX])
{

	Radix8FFT(bIsForward, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

}


#line 435 "/Engine/Private/GPUFastFourierTransformCore.ush"
groupshared float16_t SharedReal[ 2 * 2048 ];
#define NUM_BANKS 32

#line 438 "/Engine/Private/GPUFastFourierTransformCore.ush"
void CopyLocalXToGroupShared(in Complex Local[8], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
	[unroll] for (uint r = 0; r < 8; ++r, i += Stride)
	{
		uint j = i + (i / 32) * BankSkip;
		SharedReal[ j ] = Local[ r ].x;
	}
}

#line 448 "/Engine/Private/GPUFastFourierTransformCore.ush"
void CopyLocalYToGroupShared(in Complex Local[8], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
	[unroll] for (uint r = 0; r < 8; ++r, i += Stride)
	{
		uint j = i + (i / 32) * BankSkip;
		SharedReal[ j ] = Local[ r ].y;
	}
}

#line 459 "/Engine/Private/GPUFastFourierTransformCore.ush"
void CopyGroupSharedToLocalX(inout Complex Local[8], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
	[unroll] for (uint r = 0; r < 8; ++r, i += Stride) 
	{
		uint j = i + (i / 32) * BankSkip;
		Local[ r ].x = SharedReal[ j ];
	}

}

#line 471 "/Engine/Private/GPUFastFourierTransformCore.ush"
void CopyGroupSharedToLocalY(inout Complex Local[8], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
	[unroll] for (uint r = 0; r < 8; ++r, i += Stride) 
	{
		uint j = i + (i / 32) * BankSkip;
		Local[ r ].y = SharedReal[ j ];
	}

}

#line 482 "/Engine/Private/GPUFastFourierTransformCore.ush"
void CopyLocalXToGroupShared(in Complex Local[8], in uint Head, in uint Stride)
{
	CopyLocalXToGroupShared(Local, Head, Stride, 0);
}

void CopyLocalYToGroupShared(in Complex Local[8], in uint Head, in uint Stride)
{
	CopyLocalYToGroupShared(Local, Head, Stride, 0);
}
void TransposeData(inout Complex Local[RADIX], uint AHead, uint AStride, uint BHead, uint BStride)
{
	uint BankSkip = (AStride < NUM_BANKS) ? AStride : 0;

	CopyLocalXToGroupShared(Local, AHead, AStride, BankSkip);
	
	GroupMemoryBarrierWithGroupSync();
 
	CopyGroupSharedToLocalX(Local, BHead, BStride, BankSkip);

	GroupMemoryBarrierWithGroupSync();

	CopyLocalYToGroupShared(Local, AHead, AStride, BankSkip);

	GroupMemoryBarrierWithGroupSync();

	CopyGroupSharedToLocalY(Local, BHead, BStride, BankSkip);

} 
#define TWO_PI 	 2.0f * PI

void ButterflyRadix4(bool bIsForward, inout Complex Local[RADIX], uint ThreadIdx, uint Length)
{

	float AngleEven = TWO_PI * float(ThreadIdx) / float(Length * RADIX);
	float AngleOdd = TWO_PI * float(ThreadIdx+Length) / float(Length * RADIX);
	if (bIsForward)
	{
		AngleEven *= -1;
		AngleOdd *= -1;
	}

		
	Complex TwiddleIncEven;
	sincos(AngleEven, TwiddleIncEven.y, TwiddleIncEven.x);
	Complex TwiddleIncOdd;
	sincos(AngleOdd, TwiddleIncOdd.y, TwiddleIncOdd.x);
	Complex TwiddleEven = TwiddleIncEven;
	Complex TwiddleOdd = TwiddleIncOdd;
	for (uint r = 2; r < RADIX; r+=2)
	{
		Local[r] = ComplexMult(TwiddleEven, Local[r]);
		TwiddleEven = ComplexMult(TwiddleEven, TwiddleIncEven);
		Local[r+1] = ComplexMult(TwiddleOdd, Local[r+1]);
		TwiddleOdd = ComplexMult(TwiddleOdd, TwiddleIncOdd);
	}
}
void Butterfly(bool bIsForward, inout Complex Local[RADIX], uint ThreadIdx, uint Length)
{

	float angle =  TWO_PI * float( ThreadIdx % Length) / float(Length * RADIX);
	if (bIsForward) {
		angle *= -1;
	}

		
	Complex TwiddleInc;
	sincos(angle, TwiddleInc.y, TwiddleInc.x);
	Complex Twiddle = TwiddleInc;
	for (uint r = 1; r < RADIX; ++r) {
		Local[r] = ComplexMult(Twiddle, Local[r]);
		Twiddle  = ComplexMult(Twiddle, TwiddleInc);
	}
}

uint Expand(in uint j, in uint Ns, in uint R) {
	return (j / Ns) * Ns * R + (j % Ns);
}
// 对于1024 直接�?16x64 一次完�?
// 对于2048 一�?16x16 然后一�? 16x16(�?8x8复数运算pad�?8x16)
// Performs a single pass Stockham FFT using group shared memory.
void GroupSharedTCFFT(in const bool bIsForward, inout Complex Local[RADIX], in const uint ArrayLength, in const uint ThreadIdx)
{
	uint NumCols = ArrayLength / RADIX;

	//uint IdxS = Expand(j, NumCols, RADIX);
	//uint IdxS = (ThreadIdx / NumCols) * ArrayLength + (ThreadIdx % NumCols);
	
	uint IdxS = ThreadIdx;
		
    // uint Ns = 1;
	// (j / Ns) * Ns * R + (j % Ns);
	// Expand(j, Ns, RADIX);
	uint IdxD = ThreadIdx * RADIX;

	// Transform these RADIX values.
#if RADIX == 8
	Radix8FFT(bIsForward, Local);
#else
    #error "Undefined radix length!"
#endif

	
#if SCAN_LINE_LENGTH == 2048

#else
    #error "Only Support Signal Length of 2048!"
#endif

	TransposeData(Local, IdxD, 1, IdxS, NumCols);		
	// NumCols = ArrayLength / RADIX: NumCols is an int power of RADIX (ie. 0, 1, 2..)
	uint Ns = RADIX;

#if COMPILER_PSSL
	LOOP
#endif
	for ( ; Ns < NumCols; Ns *= RADIX ) 
	{
		
		Butterfly(bIsForward, Local, ThreadIdx, Ns);

		// IdxD(ThreadIdx) is contiguous blocks of lenght Ns

		IdxD = Expand(ThreadIdx, Ns, RADIX);
		//uint Expand(in uint j, in uint Ns, in uint R) { return (j / Ns) * Ns * R + (j % Ns);}

		// Transform these RADIX values.

		RadixFFT(bIsForward, Local);
		
		// Wait for the rest of the thread group to finish working on this ArrayLength of data. 

		GroupMemoryBarrierWithGroupSync();
		
		TransposeData(Local, IdxD, Ns, IdxS, NumCols);
	}

	ButterflyRadix4(bIsForward, Local, ThreadIdx, NumCols);
	
	Radix4FFT(bIsForward, Local[0], Local[2], Local[4], Local[6]);
	Radix4FFT(bIsForward, Local[1], Local[3], Local[5], Local[7]);

	// not needed for the simple case
	
	GroupMemoryBarrierWithGroupSync();

}


void GroupSharedTCFFT(in bool bIsForward, inout Complex LocalBuffer[2][RADIX], in uint ArrayLength, in uint ThreadIdx)
{
	// Note: The forward and inverse FFT require a 'normalization' scale, such that the normalization scale
	// of the forward times normalization scale of the inverse = 1 / ArrayLenght.
	// ForwardScale(ArrayLength) * InverseScale(ArrayLength) = 1 / ArrayLength;
	
	// Scale Forward
	
	if (bIsForward)
	{
#if NORMALIZE_ORDER == 1

#else
		Scale(LocalBuffer, ForwardScale(ArrayLength));
#endif
	}
	else
	{
		Scale(LocalBuffer, InverseScale(ArrayLength));
	}
	
	// Transform each buffer.
	GroupSharedTCFFT(bIsForward, LocalBuffer[1], ArrayLength, ThreadIdx);

	GroupSharedTCFFT(bIsForward, LocalBuffer[0], ArrayLength, ThreadIdx);
	


}




// groupshared float16_t SharedReal[ 2 * SCAN_LINE_LENGTH ];

void SplitTwoForOne(inout Complex LocalBuffer[RADIX], in uint Head, in uint Stride, in uint N)
{

	const uint Non2 = N / 2;	
	
	CopyLocalXToGroupShared(LocalBuffer, Head, Stride);
	
	GroupMemoryBarrierWithGroupSync();

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? ( N - K) : 0;

			float Tmp = SharedReal[NmK]; 
		
			Tmp *= (K > Non2)? -1 : 1;

			LocalBuffer[i].x += Tmp;
		}
	}

	if (Head == 0 ) LocalBuffer[0].x = 2.f * SharedReal[0];

	GroupMemoryBarrierWithGroupSync();

	CopyLocalYToGroupShared(LocalBuffer, Head, Stride);

	GroupMemoryBarrierWithGroupSync();

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? ( N - K) : 0;
			
			float Tmp = -SharedReal[NmK];
		
			Tmp *= (K < Non2)? 1 : -1;

			LocalBuffer[i].y += Tmp;
			
		}
	}

	if (Head == 0) LocalBuffer[0].y = 2.f * SharedReal[0];
 
	{
		[unroll] for (uint i = 0; i < RADIX; ++i) LocalBuffer[i] *= 0.5;
	}

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			
			if (K > Non2) LocalBuffer[i] = ComplexMult(Complex(0, -1), LocalBuffer[i] );
		}
	}
}
void SplitTwoForOne(inout Complex LocalBuffer[2][RADIX], in uint Head, in uint Stride, in uint ArrayLength)
{
	
	SplitTwoForOne(LocalBuffer[ 0 ], Head, Stride, ArrayLength);
	GroupMemoryBarrierWithGroupSync();
	 
	SplitTwoForOne(LocalBuffer[ 1 ], Head, Stride, ArrayLength);
	
}
void WriteTwoForOneFrequencyData(in bool bIsHorizontal, inout Complex LocalBuffer[2][RADIX], uint ScanIdx, uint Loc, uint Stride, uint N)
{
    RWTexture2D<float4> DstTexture = ResourceDescriptorHeap[OutputTextureOffset];

	GroupMemoryBarrierWithGroupSync();

	SplitTwoForOne(LocalBuffer, Loc, Stride, N);

	const bool bIsFirstElement = (Loc == 0);
	const uint Non2 =  N / 2;
	if (bIsHorizontal)
	{
		uint2 Pixel = uint2(Loc, ScanIdx);
		float4 DstValue;
		[unroll]
		for (uint r = 0; r < RADIX; ++r, Pixel.x += Stride)
		{
			DstValue.xy = LocalBuffer[ 0 ][ r ];
			DstValue.zw = LocalBuffer[ 1 ][ r ];
			DstTexture[Pixel] = DstValue;
			
			if (Pixel.x == Non2)
			{
				DstTexture[Pixel]                 = float4(DstValue.x, 0.f, DstValue.z, 0.f);
				DstTexture[uint2(N + 1, Pixel.y)] = float4(DstValue.y, 0.f, DstValue.w, 0.f);
			}
		}
		
		if (bIsFirstElement)
		{
			DstValue.xy = LocalBuffer[ 0 ][ 0 ];
			DstValue.zw = LocalBuffer[ 1 ][ 0 ];

			DstTexture[uint2(0, Pixel.y)]  = float4(DstValue.x, 0.f, DstValue.z, 0.f); 
			DstTexture[uint2(N, Pixel.y)]  = float4(DstValue.y, 0.f, DstValue.w, 0.f); 
		}
	}
	else
	{
		uint2 Pixel = uint2(ScanIdx, Loc);
		float4 DstValue;
		[unroll]
		for (uint r = 0; r < RADIX; ++r, Pixel.y += Stride)
		{
			
			DstValue.xy = LocalBuffer[ 0 ][ r ];
			DstValue.zw = LocalBuffer[ 1 ][ r ];
			DstTexture[Pixel] = DstValue;
			
			if (Pixel.y == Non2)
			{
				DstTexture[Pixel]                 = float4(DstValue.x, 0.f, DstValue.z, 0.f);
				DstTexture[uint2(Pixel.x, N + 1)] = float4(DstValue.y, 0.f, DstValue.w, 0.f);
			}
		
		}
		
		if (bIsFirstElement)
		{
			DstValue.xy = LocalBuffer[ 0 ][ 0 ];
			DstValue.zw = LocalBuffer[ 1 ][ 0 ];

			DstTexture[uint2(Pixel.x, 0)]  = float4(DstValue.x, 0.f, DstValue.z, 0.f); 
			DstTexture[uint2(Pixel.x, N)]  = float4(DstValue.y, 0.f, DstValue.w, 0.f); 
		}
	}
}



void CopyDataLocalToDstWindow(in Complex LocalBuffer[2][RADIX], bool bIsHorizontal, in uint ScanIdx, uint Loc, uint Stride, uint4 ROIRect)
{
    RWTexture2D<float4> DstTexture = ResourceDescriptorHeap[OutputTextureOffset];

	if(bIsHorizontal)
	{
		
		uint2 Pixel = uint2(Loc + ROIRect.x, ScanIdx + ROIRect.y);

		[unroll]
		for (uint r = 0; r < RADIX && Pixel.x < ROIRect.z; ++r, Pixel.x += Stride)
		{
			float4 DstValue;
			DstValue.xy = LocalBuffer[0][r];
			DstValue.zw = LocalBuffer[1][r];

			DstTexture[Pixel] = DstValue;
		}
	}
	else
	{
		uint2 Pixel = uint2(ScanIdx + ROIRect.x, Loc + ROIRect.y);

		[unroll]
		for (uint r = 0; r < RADIX && Pixel.y < ROIRect.w; ++r, Pixel.y += Stride)
		{
			float4 DstValue;
			DstValue.xy = LocalBuffer[0][r];
			DstValue.zw = LocalBuffer[1][r];

			DstTexture[Pixel] = DstValue;
		}
	}
}
void CopyDataLocalToDstWindow(in Complex LocalBuffer[2][RADIX], bool bIsHorizontal, uint ScanIdx, uint Loc, uint Stride, uint2 Extent)
{
	uint4 ROIRect = uint4(0, 0, Extent.x, Extent.y);
	CopyDataLocalToDstWindow(LocalBuffer, bIsHorizontal, ScanIdx, Loc, Stride, ROIRect);
}

void MergeTwoForOne(inout Complex LocalBuffer[RADIX], in uint Head, in uint Stride, in uint N)
{
	
	uint Non2 = N / 2;

	float TmpX[RADIX];
	{
		for (uint i = 0; i < RADIX; ++i) TmpX[i] = LocalBuffer[i].x;
	}

	CopyLocalYToGroupShared(LocalBuffer, Head, Stride);
	
	GroupMemoryBarrierWithGroupSync();
	
	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? (N - K) : 0 ;
	
			float Tmp = SharedReal[ NmK ]; 
			Tmp *= (K > Non2) ? -1 : 1;
		
			LocalBuffer[i].x += Tmp;
		}
	}
	
	Complex FirstElement  = Complex(0, SharedReal[0]);
	Complex MiddleElement = Complex(0, SharedReal[Non2]);

	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint r = 0, i = Head; r < RADIX; ++r, i += Stride)
	{
		SharedReal[ i ] = TmpX[ r ];
	}

	GroupMemoryBarrierWithGroupSync();
	FirstElement.x  = SharedReal[0];
	MiddleElement.x = SharedReal[Non2];

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? (N - K) : 0 ;
			
			float Tmp = SharedReal[ NmK ]; 
			Tmp *= (K > Non2) ? -1 : 1;
		
			LocalBuffer[i].y += Tmp;
		}
	}

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			
			if (K > Non2) LocalBuffer[ i ] = ComplexMult(Complex(0, 1), LocalBuffer[ i ]);

			if (K == Non2)
			{	
				
				LocalBuffer[ i ] = MiddleElement;
			}
		}
	}

	if (Head == 0) LocalBuffer[ 0 ] = FirstElement;

}

void MergeTwoForOne(inout Complex LocalBuffer[2][RADIX], in uint Head, in uint Stride, in uint ArrayLength)
{
	
    MergeTwoForOne(LocalBuffer[ 0 ], Head, Stride, ArrayLength);
    GroupMemoryBarrierWithGroupSync();
    
    MergeTwoForOne(LocalBuffer[ 1 ], Head, Stride, ArrayLength);
	 
}

void ReadTwoForOneFrequencyData(bool bIsHorizontal, inout Complex LocalBuffer[2][RADIX], in uint ScanIdx, in uint Loc, in uint Stride, in uint N)
{
	const bool bIsFirstElement = (Loc == 0);
	const uint Non2 =  N / 2;
    Texture2D SrcTexture = ResourceDescriptorHeap[InputTextureOffset];

	if (bIsHorizontal) 
	{
		// last two values
		float4 NValue   = SrcTexture[uint2(N, ScanIdx)];
		float4 NppValue = SrcTexture[uint2(N +1, ScanIdx)];

		uint2 Pixel = uint2(Loc, ScanIdx);
		[unroll]
		for (uint i = 0; i < RADIX; ++i, Pixel.x += Stride)
		{	
			float4 SrcValue = SrcTexture[Pixel];
			LocalBuffer[ 0 ][ i ] = SrcValue.xy;
			LocalBuffer[ 1 ][ i ] = SrcValue.zw;

			if ( Pixel.x ==  Non2)
			{
				// local buffer will be pure real with F_N/2,  need to add I * G_N/2 (G_N/2 is real ie float2(G_r, 0))
				float4 TmpValue = NppValue; // will be (#,0,#,0)
				LocalBuffer[ 0 ][ i ] += NppValue.yx;
				LocalBuffer[ 1 ][ i ] += NppValue.wz;
			}

		}

		if (bIsFirstElement)
		{
			float4 LastSrcValue = SrcTexture[uint2(N, Pixel.y)]; // will be (#,0,#,0)
			LocalBuffer[ 0 ][ 0 ] += NValue.yx; 
			LocalBuffer[ 1 ][ 0 ] += NValue.wz;
		}

	}
	else 
	{
		// last two values
	    float4 NValue   = SrcTexture[uint2(ScanIdx, N)];
		float4 NppValue = SrcTexture[uint2(ScanIdx, N + 1)];
		
		uint2 Pixel = uint2(ScanIdx, Loc);
		[unroll]
		for (uint i = 0; i < RADIX; ++i, Pixel.y += Stride)
		{
			float4 SrcValue = SrcTexture[Pixel];
			LocalBuffer[ 0 ][ i ] = SrcValue.xy;
			LocalBuffer[ 1 ][ i ] = SrcValue.zw;

			if ( Pixel.y ==  Non2)
			{
				// local buffer will be pure real with F_N/2,  need to add IG_N/2
				LocalBuffer[ 0 ][ i ] += NppValue.yx;
				LocalBuffer[ 1 ][ i ] += NppValue.wz;
			}
		}
	
		if (bIsFirstElement)
		{
			LocalBuffer[ 0 ][ 0 ] += NValue.yx; 
			LocalBuffer[ 1 ][ 0 ] += NValue.wz;
		}
	}

	// Combine the transforms of the two real signals (F,G) as Z = F + I G
	MergeTwoForOne(LocalBuffer, Loc, Stride, N);
	
	// Done with the group shared memory that was used in the merge
	GroupMemoryBarrierWithGroupSync();
}
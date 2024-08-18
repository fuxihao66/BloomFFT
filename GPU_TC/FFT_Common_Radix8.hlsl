#define RADIX 8
#define Complex float16_t2

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


groupshared float16_t groupMatInput[3 * 2048];
// groupshared float16_t groupMatF[256 * 2];
groupshared float16_t groupMatOutput[2* 2048];

static const float PI = 3.14159265f;

// 对于1024 直接用16x64 一次完成
// 对于2048 一次16x16 然后一次 16x16(把8x8复数运算pad成8x16)
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
	ByteAddressBuffer FBuffer = bIsForward ? ResourceDescriptorHeap[FBufferOffset] : ResourceDescriptorHeap[FBufferInverseOffset];

    WaveMatrixLeft<float16_t, 16, 16> mat_f_real;
    WaveMatrixLeft<float16_t, 16, 16> mat_f_imag;

    mat_f_real.Load(FBuffer, 						    0, 16 * /*sizeof(float16_t)*/2, false);
    mat_f_imag.Load(FBuffer, 256 * /*sizeof(float16_t)*/2, 16 * /*sizeof(float16_t)*/2, false);

	const uint WaveIndex = WaveReadLaneFirst(IdxS / WAVE_SIZE);
	const uint groupsize = SCAN_LINE_LENGTH / RADIX;
	// const uint finalMergeNum = groupsize / 16;
	{
		WaveMatrixRight<float16_t, 16, 16> mat_x; // x for real part
		WaveMatrixRight<float16_t, 16, 16> mat_y;// y for imag part
		
		WaveMatrixAccumulator<float16_t, 16, 16> result;
		// element wise wultiplication of twiddle
		// const int rowIndex = IdxS / finalMergeNum;
		[unroll]
		for (int j = 0; j < RADIX; j++){// TODO: remove bank conflict
			float expon = -2.f * PI * j * rowIndex / (256.f);
			if (!bIsForward)
				expon *= -1.f;
			Complex TwiddleScaled = Complex(cos(expon), sin(expon));
			Complex ElementWiseResult = ComplexMult(TwiddleScaled, Local[j]);
			groupMatInput[RADIX * 3 * IdxS + j] = - ElementWiseResult.y;
			groupMatInput[RADIX * 3 * IdxS + j + RADIX] = ElementWiseResult.x;
			groupMatInput[RADIX * 3 * IdxS + j + RADIX * 2] = ElementWiseResult.y;
		}
		GroupMemoryBarrierWithGroupSync();

		[unroll]
		for (int i = 0; i < 2; i++){// each wave deals with two 16x16 complex multiplication
			mat_x.Load(groupMatInput, (WaveIndex * 2 + i) * 24 + 8, 3 * SCAN_LINE_LENGTH / 16, false);// TODO: remove bank conflict
			mat_y.Load(groupMatInput, (WaveIndex * 2 + i) * 24, 3 * SCAN_LINE_LENGTH / 16, false);
			
			result.Multiply(mat_f_real, mat_x);
			result.MultiplyAccumulate(mat_f_imag, mat_y);

			result.Store(groupMatOutput, (WaveIndex * 2 + i) * 256, 16, false);// real
		}
	}	

	GroupMemoryBarrierWithGroupSync();

	{
		WaveMatrixRight<float16_t, 16, 16> mat_x; // x for real part
		WaveMatrixRight<float16_t, 16, 16> mat_y;// y for imag part
		WaveMatrixAccumulator<float16_t, 16, 16> result_Ab;
		WaveMatrixAccumulator<float16_t, 16, 16> result_Bb;

		// element wise
		{
			const uint stride = SCAN_LINE_LENGTH / RADIX * 2;

			for (int i = 0; i < RADIX / 2; i++){// TODO: merge twiddle to previous stage?
				for (int j = 0; j < 2; j++){
					uint FlattenedIndex = stride * i + 2 * IdxS + j;
					const Complex v = Complex(groupMatOutput[FlattenedIndex], groupMatOutput[FlattenedIndex + SCAN_LINE_LENGTH]);
					const uint colIndex = FlattenedIndex % 256;
					const uint rowIndex = FlattenedIndex / 256;
					float expon = -2.f * PI * colIndex * rowIndex / float16_t(SCAN_LINE_LENGTH);
					if (!bIsForward){
						expon *= -1.f;
					}
					const Complex Twiddle = Complex(cos(expon), sin(expon));
					const Complex Result = ComplexMult(v, Twiddle);
					groupMatOutput[FlattenedIndex] = Result.x;
					groupMatOutput[FlattenedIndex + SCAN_LINE_LENGTH] = Result.y;
				}
			}
		}

		GroupMemoryBarrierWithGroupSync();

		{
			mat_x.Load(groupMatOutput, 2 * i + WaveIndex * 64, 256, false);
			mat_y.Load(groupMatOutput, 2 * i + WaveIndex * 64 + SCAN_LINE_LENGTH, 256, false);

			result_Ab.Multiply(mat_f_real, mat_y);
			result_Bb.Multiply(mat_f_imag, mat_y);

			result_Bb.ScalarMultiply((float16_t)(-1));
			result_Bb.MultiplyAccumulate(mat_f_real, mat_x);
			result_Ab.MultiplyAccumulate(mat_f_imag, mat_x);

			result_Bb.Store(groupMatInput, 2 * i + WaveIndex * 64, 128, false);// real
			result_Ab.Store(groupMatInput, 2 * i + WaveIndex * 64 + SCAN_LINE_LENGTH, 128, false);// imag
		}
	}
	
	GroupMemoryBarrierWithGroupSync();

	// write result back to local buffer
	for (int i = 0; i < RADIX; i++){
		Local[i] = Complex(groupMatInput[i * groupsize + IdxS], groupMatInput[i * groupsize + IdxS + SCAN_LINE_LENGTH]);
	}

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
#define NUM_BANKS 32

void CopyLocalXToGroupShared(in Complex Local[RADIX], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
    [unroll]
	for (uint r = 0; r < RADIX; ++r, i += Stride)
	{
		uint j = i + (i / NUM_BANKS) * BankSkip;
		groupMatInput[ j ] = Local[ r ].x;
	}
}

void CopyLocalYToGroupShared(in Complex Local[RADIX], in uint Head, in uint Stride, in uint BankSkip)
{
	uint i = Head;
	[unroll]
    for (uint r = 0; r < RADIX; ++r, i += Stride)
	{
		uint j = i + (i / NUM_BANKS) * BankSkip;
		groupMatInput[ j ] = Local[ r ].y;
	}
}

void CopyLocalXToGroupShared(in Complex Local[RADIX], in uint Head, in uint Stride)
{
	CopyLocalXToGroupShared(Local, Head, Stride, 0);
}

void CopyLocalYToGroupShared(in Complex Local[RADIX], in uint Head, in uint Stride)
{
	CopyLocalYToGroupShared(Local, Head, Stride, 0);
}
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

			float Tmp = groupMatInput[NmK]; 
		
			Tmp *= (K > Non2)? -1 : 1;

			LocalBuffer[i].x += Tmp;
		}
	}

	if (Head == 0 ) LocalBuffer[0].x = 2.f * groupMatInput[0];

	GroupMemoryBarrierWithGroupSync();

	CopyLocalYToGroupShared(LocalBuffer, Head, Stride);

	GroupMemoryBarrierWithGroupSync();

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? ( N - K) : 0;
			
			float Tmp = -groupMatInput[NmK];
		
			Tmp *= (K < Non2)? 1 : -1;

			LocalBuffer[i].y += Tmp;
			
		}
	}

	if (Head == 0) LocalBuffer[0].y = 2.f * groupMatInput[0];
 
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
	
			float Tmp = groupMatInput[ NmK ]; 
			Tmp *= (K > Non2) ? -1 : 1;
		
			LocalBuffer[i].x += Tmp;
		}
	}
	
	Complex FirstElement  = Complex(0, groupMatInput[0]);
	Complex MiddleElement = Complex(0, groupMatInput[Non2]);

	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint r = 0, i = Head; r < RADIX; ++r, i += Stride)
	{
		groupMatInput[ i ] = TmpX[ r ];
	}

	GroupMemoryBarrierWithGroupSync();
	FirstElement.x  = groupMatInput[0];
	MiddleElement.x = groupMatInput[Non2];

	{
		[unroll]
		for (uint i = 0, K = Head; i < RADIX; ++i, K += Stride)
		{
			uint NmK = (K > 0) ? (N - K) : 0 ;
			
			float Tmp = groupMatInput[ NmK ]; 
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
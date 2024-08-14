.\dxc_preview\bin\x64\dxc.exe /D"SCAN_LINE_LENGTH=1024" /Zi /E"CSMain" /Vn"g_TwoForOneShader_1024_CS" /HV"2018" /enable-16bit-types /Tcs_6_8 /Fh"FFT_TwoForOne_1024.hlsl.h" /nologo "./FFT_TwoForOne.hlsl"
.\dxc_preview\bin\x64\dxc.exe /D"SCAN_LINE_LENGTH=2048" /Zi /E"CSMain" /Vn"g_TwoForOneShader_2048_CS" /HV"2018" /enable-16bit-types /Tcs_6_8 /Fh"FFT_TwoForOne_2048.hlsl.h" /nologo "./FFT_TwoForOne.hlsl"
.\dxc_preview\bin\x64\dxc.exe /D"SCAN_LINE_LENGTH=1024" /Zi /E"GSConvolutionWithTextureCS" /Vn"g_ConvWithTextureShader_1024_CS" /HV"2018" /enable-16bit-types /Tcs_6_8 /Fh"FFT_ConvWithTexture_1024.hlsl.h" /nologo "./FFT_ConvWithTexture.hlsl"
.\dxc_preview\bin\x64\dxc.exe /D"SCAN_LINE_LENGTH=2048" /Zi /E"GSConvolutionWithTextureCS" /Vn"g_ConvWithTextureShader_2048_CS" /HV"2018" /enable-16bit-types /Tcs_6_8 /Fh"FFT_ConvWithTexture_2048.hlsl.h" /nologo "./FFT_ConvWithTexture.hlsl"


@REM https://strontic.github.io/xcyclopedia/library/dxc.exe-0C1709D4E1787E3EB3E6A35C85714824.html
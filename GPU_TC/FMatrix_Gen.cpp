// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <Windows.h>

class Float16Compressor
{
	union Bits
	{
		float f;
		int32_t si;
		uint32_t ui;
	};

	static int const shift = 13;
	static int const shiftSign = 16;

	static int32_t const infN = 0x7F800000; // flt32 infinity
	static int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
	static int32_t const minN = 0x38800000; // min flt16 normal as a flt32
	static int32_t const signN = 0x80000000; // flt32 sign bit

	static int32_t const infC = infN >> shift;
	static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
	static int32_t const maxC = maxN >> shift;
	static int32_t const minC = minN >> shift;
	static int32_t const signC = signN >> shiftSign; // flt16 sign bit

	static int32_t const mulN = 0x52000000; // (1 << 23) / minN
	static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

	static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
	static int32_t const norC = 0x00400; // min flt32 normal down shifted

	static int32_t const maxD = infC - maxC - 1;
	static int32_t const minD = minC - subC - 1;

public:

	static uint16_t compress(float value)
	{
		Bits v, s;
		v.f = value;
		uint32_t sign = v.si & signN;
		v.si ^= sign;
		sign >>= shiftSign; // logical shift
		s.si = mulN;
		s.si = static_cast<int32_t>(s.f * v.f); // correct subnormals
		v.si ^= (s.si ^ v.si) & -(minN > v.si);
		v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
		v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
		v.ui >>= shift; // logical shift
		v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
		v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
		return static_cast<uint16_t>(v.ui | sign);
	}

	static float decompress(uint16_t value)
	{
		Bits v;
		v.ui = value;
		int32_t sign = v.si & signC;
		v.si ^= sign;
		sign <<= shiftSign;
		v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
		v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
		Bits s;
		s.si = mulC;
		s.f *= v.si;
		int32_t mask = -(norC > v.si);
		v.si <<= shift;
		v.si ^= (s.si ^ v.si) & mask;
		v.si |= sign;
		return v.f;
	}
};
#define PI 3.14159265f
int main()
{
	uint16_t										FBufferData_1024[256 * 4];
	uint16_t										FBufferData_Inverse_1024[256 * 4];
	uint16_t										FBufferData_2048[256 * 4];
	uint16_t										FBufferData_Inverse_2048[256 * 4];

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			UINT FlattenedIndex = i * 16 + j;

			float Expo = -2.f * PI * float(i) * float(j) / 16.f;
			FBufferData_1024[FlattenedIndex] = Float16Compressor::compress(std::cos(Expo));
			FBufferData_1024[FlattenedIndex + 256] = Float16Compressor::compress(std::sin(Expo));
			FBufferData_Inverse_1024[FlattenedIndex] = Float16Compressor::compress(std::cos(-Expo));
			FBufferData_Inverse_1024[FlattenedIndex + 256] = Float16Compressor::compress(std::sin(-Expo));

			FBufferData_2048[FlattenedIndex] = FBufferData_1024[FlattenedIndex];
			FBufferData_2048[FlattenedIndex + 256] = FBufferData_1024[FlattenedIndex + 256];
			FBufferData_Inverse_2048[FlattenedIndex] = FBufferData_Inverse_1024[FlattenedIndex];
			FBufferData_Inverse_2048[FlattenedIndex + 256] = FBufferData_Inverse_1024[FlattenedIndex + 256];
		}
	}
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			UINT FlattenedIndex = i * 16 + j;
			if (i >= 4 || j >= 4) {
				FBufferData_1024[FlattenedIndex + 512] = Float16Compressor::compress(0.f);
				FBufferData_1024[FlattenedIndex + 768] = Float16Compressor::compress(0.f);
				FBufferData_Inverse_1024[FlattenedIndex + 512] = Float16Compressor::compress(0.f);
				FBufferData_Inverse_1024[FlattenedIndex + 768] = Float16Compressor::compress(0.f);
			}
			else {
				float Expo = -2.f * PI * float(i) * float(j) / 4.f;

				FBufferData_1024[FlattenedIndex + 512] = Float16Compressor::compress(std::cos(Expo));
				FBufferData_1024[FlattenedIndex + 768] = Float16Compressor::compress(std::sin(Expo));
				FBufferData_Inverse_1024[FlattenedIndex + 512] = Float16Compressor::compress(std::cos(-Expo));
				FBufferData_Inverse_1024[FlattenedIndex + 768] = Float16Compressor::compress(std::sin(-Expo));
			}
			if (i >= 8 || j >= 8) {
				FBufferData_2048[FlattenedIndex + 512] = Float16Compressor::compress(0.f);
				FBufferData_2048[FlattenedIndex + 768] = Float16Compressor::compress(0.f);
				FBufferData_Inverse_2048[FlattenedIndex + 512] = Float16Compressor::compress(0.f);
				FBufferData_Inverse_2048[FlattenedIndex + 768] = Float16Compressor::compress(0.f);
			}
			else {
				float Expo = -2.f * PI * float(i) * float(j) / 8.f;

				FBufferData_2048[FlattenedIndex + 512] = Float16Compressor::compress(std::cos(Expo));
				FBufferData_2048[FlattenedIndex + 768] = Float16Compressor::compress(std::sin(Expo));
				FBufferData_Inverse_2048[FlattenedIndex + 512] = Float16Compressor::compress(std::cos(-Expo));
				FBufferData_Inverse_2048[FlattenedIndex + 768] = Float16Compressor::compress(std::sin(-Expo));
			}
		}
	}
	{
		std::ofstream myfile;
		myfile.open("f_1024.txt");
		myfile.write(reinterpret_cast<const char*>(FBufferData_1024), std::streamsize(256 * 4 * sizeof(uint16_t)));
		myfile.close();
	}
	{
		std::ofstream myfile;
		myfile.open("f_inv_1024.txt");
		myfile.write(reinterpret_cast<const char*>(FBufferData_Inverse_1024), std::streamsize(256 * 4 * sizeof(uint16_t)));
		myfile.close();
	}
	{
		std::ofstream myfile;
		myfile.open("f_2048.txt");
		myfile.write(reinterpret_cast<const char*>(FBufferData_2048), std::streamsize(256 * 4 * sizeof(uint16_t)));
		myfile.close();
	}
	{
		std::ofstream myfile;
		myfile.open("f_inv_2048.txt");
		myfile.write(reinterpret_cast<const char*>(FBufferData_Inverse_2048), std::streamsize(256 * 4 * sizeof(uint16_t)));
		myfile.close();
	}

	/*{
		uint16_t test[256 * 4];
		std::ifstream myfile;
		myfile.open("f_1024.txt");
		myfile.read(reinterpret_cast<char*>(test), std::streamsize(256 * 4 * sizeof(uint16_t)));
		myfile.close();

		std::cout << test[0];
	}*/
	

    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

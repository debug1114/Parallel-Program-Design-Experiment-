#include<iostream>
#include<arm_neon.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<tmmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<immintrin.h>
#include <windows.h>


using namespace std;

const int N = 4096;

//设置地址对齐
__declspec(align(16)) float A[N][N];
__declspec(align(16)) float B[N][N];

//打印
void print(float A[N][N], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			cout << A[i][j] << ",";
		cout << endl;
	}
}

//普通高斯消元
void Gauss_ordinary(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}


//SSE指令高斯消元
void Gauss_SIMD_SSE(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		__m128 vt = _mm_set_ps1(A[k][k]);
		int j;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			__m128 va = _mm_load_ps(A[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(A[k] + j, va);
		}
		for (; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			__m128 vaik = _mm_set_ps1(A[i][k]);
			for (j = k + 1; j + 3 < n; j += 4)
			{
				__m128 vakj = _mm_load_ps(A[k] + j);
				__m128 vaij = _mm_load_ps(A[i] + j);
				__m128 vx = _mm_mul_ps(vaik, vakj);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(A[i] + j, vaij);
			}
			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
	//print(A, N);
}

//AVX指令高斯消元
void Gauss_SIMD_AVX(float A[N][N],int n)
{
	for (int k = 0; k < n; k++)
	{
		__m256 vt = _mm256_set1_ps(A[k][k]);
		int j;
		for (j = k + 1; j + 7 < n; j += 8)
		{
			__m256 va = _mm256_load_ps(A[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(A[k] + j, va);
		}
		for (; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			__m256 vaik = _mm256_set1_ps(A[i][k]);
			for (j = k + 1; j + 7 < n; j += 8)
			{
				__m256 vakj = _mm256_load_ps(A[k] + j);
				__m256 vaij = _mm256_load_ps(A[i] + j);
				__m256 vx = _mm256_mul_ps(vaik, vakj);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(A[i] + j, vaij);
			}
			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
	//print(A, N);
}

//Neon指令高斯消元
void Gauss_SIMD_Neon(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		float32x4_t vt = vmovq_n_f32(A[k][k]);
		int j;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			float32x4_t va = vld1q_f32(A[k] + j);
			va = vdivq_f32(va, vt);
			vst1q_f32(A[k] + j, va);
		}
		for (; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			float32x4_t vaik = vmovq_n_f32(A[i][k]);
			for (j = k + 1; j + 3 < n; j += 4)
			{
				float32x4_t vakj = vld1q_f32(A[k] + j);
				float32x4_t vaij = vld1q_f32(A[i] + j);
				float32x4_t vx = vmulq_f32(vaik, vakj);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(A[i] + j, vaij);
			}
			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
	//print(A, N);
}


//N阶上三角阵（每个元素均为1）
void generate_up(float B[N][N], int n)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			B[i][j] = 1;
		}
	}
	//print(B, n);
}

//生成待高斯消元的矩阵
void generate_gauss(float A[N][N], float B[N][N], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			A[i][j] = B[i][j];
	}
	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n; j++)
			A[i + 1][j] = A[i][j] + A[i + 1][j];
	}
	//print(A, n);
}

//判断
bool is_right(float A[N][N], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (A[i][j] != B[i][j])
				return false;
		}
	}
	return true;
}

int main()
{


	generate_up(B, N);
	generate_gauss(A, B, N);

	long long head, tail, freq;  //timers

	QueryPerformanceFrequency((LARGE_INTEGER *) & freq);

	QueryPerformanceCounter((LARGE_INTEGER *) & head);
	for (int i = 0; i < 1; i++)
		Gauss_SIMD_AVX(A, N);
	QueryPerformanceCounter((LARGE_INTEGER *) & tail);


	cout << "Time: " << ((tail - head) * 1000.0 / freq) / 1.0 << "ms" << endl;

	if (is_right(A, N))
	{
		cout << "It is right!" << endl;
	}
	else
	{
		cout << "It is wrong!" << endl;
	}
	return 0;

}

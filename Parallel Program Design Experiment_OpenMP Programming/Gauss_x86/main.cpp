#include<iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<tmmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include <nmmintrin.h>
#include <omp.h>

using namespace std;

//ȫ�ֱ���
int N;
int step[5] = { 16,64,256,1024,4096 };
//���õ�ַ�������
__declspec(align(16))float** Gauss;//����Ԫ�ľ���
__declspec(align(16))float** UP;//�����Ǿ���
//�߳���
const int thread_count = 4;

//������

//N����������ÿ��Ԫ�ؾ�Ϊ1��
void generate_up(int n)
{
	N = n;
	UP = new float* [N];
	for (int i = 0; i < N; i++)
		UP[i] = new float[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			UP[i][j] = 1;
		}
	}
	//print(UP, n);
}

//���ɴ���˹��Ԫ�ľ���
void generate_gauss()
{
	Gauss = new float* [N];
	for (int i = 0; i < N; i++)
		Gauss[i] = new float[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			Gauss[i][j] = UP[i][j];
	}
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = 0; j < N; j++)
			Gauss[i + 1][j] = Gauss[i][j] + Gauss[i + 1][j];
	}
	//print(A, n);
}

//�ж�
bool is_right()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (Gauss[i][j] != UP[i][j])
				return false;
		}
	}
	return true;
}

//��ӡ
void print(float** A, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			cout << A[i][j] << ",";
		cout << endl;
	}
}

//�����㷨:��ͨ��˹��Ԫ
void Gauss_ordinary()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//SSEָ���˹��Ԫ
void Gauss_SIMD_SSE()
{
	for (int k = 0; k < N; k++)
	{
		__m128 vt = _mm_set_ps1(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			__m128 va = _mm_load_ps(Gauss[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			__m128 vaik = _mm_set_ps1(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				__m128 vakj = _mm_load_ps(Gauss[k] + j);
				__m128 vaij = _mm_load_ps(Gauss[i] + j);
				__m128 vx = _mm_mul_ps(vaik, vakj);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

//AVXָ���˹��Ԫ
void Gauss_SIMD_AVX()
{
	for (int k = 0; k < N; k++)
	{
		__m256 vt = _mm256_set1_ps(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 7 < N; j += 8)
		{
			__m256 va = _mm256_load_ps(Gauss[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			__m256 vaik = _mm256_set1_ps(Gauss[i][k]);
			for (j = k + 1; j + 7 < N; j += 8)
			{
				__m256 vakj = _mm256_load_ps(Gauss[k] + j);
				__m256 vaij = _mm256_load_ps(Gauss[i] + j);
				__m256 vx = _mm256_mul_ps(vaik, vakj);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}


//omp_static�汾��˹��Ԫ
void Gauss_omp_static()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(static)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_dynamic�汾��˹��Ԫ
void Gauss_omp_dynamic()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(dynamic,24)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic,24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_guided�汾��˹��Ԫ
void Gauss_omp_guided()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(guided,24)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(guided,24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_SSE�汾��˹��Ԫ
void Gauss_SSE_omp()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
		__m128 vt = _mm_set_ps1(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			__m128 va = _mm_load_ps(Gauss[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(Gauss[k] + j, va);
		}
#pragma omp single
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			__m128 vaik = _mm_set_ps1(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				__m128 vakj = _mm_load_ps(Gauss[k] + j);
				__m128 vaij = _mm_load_ps(Gauss[i] + j);
				__m128 vx = _mm_mul_ps(vaik, vakj);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

//omp_AVX�汾��˹��Ԫ
void Gauss_AVX_omp()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
		__m256 vt = _mm256_set1_ps(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 7 < N; j += 8)
		{
			__m256 va = _mm256_load_ps(Gauss[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(Gauss[k] + j, va);
		}
#pragma omp single
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			__m256 vaik = _mm256_set1_ps(Gauss[i][k]);
			for (j = k + 1; j + 7 < N; j += 8)
			{
				__m256 vakj = _mm256_load_ps(Gauss[k] + j);
				__m256 vaij = _mm256_load_ps(Gauss[i] + j);
				__m256 vx = _mm256_mul_ps(vaik, vakj);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}


int main()
{
	for (int i = 0; i < 5; i++)
	{
		long long head, tail, freq;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		float time = 0;
		generate_up(step[i]);
		cout << "N: " << N << endl;

		//���и�˹��Ԫ
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_ordinary();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "���и�˹��Ԫ:" << time << "ms" << endl;

		//SSE�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_SIMD_SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "SSE�汾��˹��Ԫ:" << time << "ms" << endl;

		//AVX�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_SIMD_AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "AVX�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_static�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_omp_static();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_static�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_dynamic�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_omp_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_dynamic�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_guided�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_omp_guided();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_guided�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_SSE�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_SSE_omp();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_SSE�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_AVX�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_AVX_omp();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_AVX�汾��˹��Ԫ:" << time << "ms" << endl;

		cout << endl;
	}
}

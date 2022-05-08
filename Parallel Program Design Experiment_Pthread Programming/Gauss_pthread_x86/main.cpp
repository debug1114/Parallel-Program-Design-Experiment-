#define HAVE_STRUCT_TIMESPEC
#pragma comment(lib, "pthreadVC2.lib")

#include <iostream>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<tmmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include <nmmintrin.h>
#include <pthread.h>
#include <semaphore.h>
#include<windows.h>
#include <stdlib.h>
#include<iomanip>

#include"prepthread.h"

using namespace std;

int main()
{
	for (int i = 0; i < 3; i++)
	{
		long long head, tail, freq;
		float seconds;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		float time = 0;
		generate_up(Step[i]);
		cout << "N: " << N << endl;

		if (N < 500)
		{
			//串行普通高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_ordinary();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "Gauss_ordinary:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//SSE高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_SIMD_SSE();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "Gauss_SIMD_SSE:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//AVX高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_SIMD_AVX();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "Gauss_SIMD_AVX:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread两重，信号量
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread2_sem();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread2_sem:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread三重，信号量
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread3_sem();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread3_sem:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread三重，barrier
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread3_barrier();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread3_barrier:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread动态线程
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread_dynamic();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread_dynamic:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread+SSE
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread2_sem_SSE();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread2_sem_SSE:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;

			//pthread+AVX
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			for (int j = 0; j < 100; j++)
				Gauss_pthread2_sem_AVX();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			//cout << "pthread2_sem_AVX:" << time << "ms" << endl;
			cout << time / 100 << "ms" << endl;
		}
		else
		{
			//串行普通高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_ordinary();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "Gauss_ordinary:" << time << "ms" << endl;

			//SSE高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_SIMD_SSE();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "Gauss_SIMD_SSE:" << time << "ms" << endl;

			//AVX高斯消元
			time = 0;
			generate_gauss();
			//print_matrix();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_SIMD_AVX();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "Gauss_SIMD_AVX:" << time << "ms" << endl;

			//pthread两重，信号量
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread2_sem();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread2_sem:" << time << "ms" << endl;

			//pthread三重，信号量
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread3_sem();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread3_sem:" << time << "ms" << endl;

			//pthread三重，barrier
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread3_barrier();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread3_barrier:" << time << "ms" << endl;

			//pthread动态线程
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread_dynamic();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread_dynamic:" << time << "ms" << endl;

			//pthread+SSE
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread2_sem_SSE();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread2_sem_SSE:" << time << "ms" << endl;

			//pthread+AVX
			time = 0;
			generate_gauss();
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
			Gauss_pthread2_sem_AVX();
			QueryPerformanceCounter((LARGE_INTEGER*)&tail);
			time = (tail - head) * 1000.0 / freq;
			cout << "pthread2_sem_AVX:" << time << "ms" << endl;
		}
	}
}


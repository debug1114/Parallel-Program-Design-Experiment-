#include<iostream>
#include <sys/time.h>
#include <omp.h>
#include<arm_neon.h>

using namespace std;

//ȫ�ֱ���
int N;
int step[5] = { 16,64,256,1024,4096 };
float** Gauss;//����Ԫ�ľ���
float** UP;//�����Ǿ���
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

//Neonָ���˹��Ԫ
void Gauss_SIMD_Neon()
{
	for (int k = 0; k < N; k++)
	{
		float32x4_t vt = vmovq_n_f32(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			float32x4_t va = vld1q_f32(Gauss[k] + j);
			va = vdivq_f32(va, vt);
			vst1q_f32(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			float32x4_t vaik = vmovq_n_f32(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				float32x4_t vakj = vld1q_f32(Gauss[k] + j);
				float32x4_t vaij = vld1q_f32(Gauss[i] + j);
				float32x4_t vx = vmulq_f32(vaik, vakj);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(A, N);
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

//omp_Neonָ���˹��Ԫ
void Gauss_Neon_omp()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
		float32x4_t vt = vmovq_n_f32(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			float32x4_t va = vld1q_f32(Gauss[k] + j);
			va = vdivq_f32(va, vt);
			vst1q_f32(Gauss[k] + j, va);
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
			float32x4_t vaik = vmovq_n_f32(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				float32x4_t vakj = vld1q_f32(Gauss[k] + j);
				float32x4_t vaij = vld1q_f32(Gauss[i] + j);
				float32x4_t vx = vmulq_f32(vaik, vakj);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(A, N);
}

int main()
{
	for (int i = 0; i < 5; i++)
	{
		struct timeval start;
		struct timeval end;
		float time = 0;
		generate_up(step[i]);
		cout << "N: " << N << endl;

		//���и�˹��Ԫ
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_ordinary();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "���и�˹��Ԫ:" << time << "ms" << endl;


		//omp_static�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_omp_static();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "omp_static�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_dynamic�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_omp_dynamic();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "omp_dynamic�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_guided�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_omp_guided();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "omp_guided�汾��˹��Ԫ:" << time << "ms" << endl;

		//omp_Neon�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_Neon_omp();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "omp_Neon�汾��˹��Ԫ:" << time << "ms" << endl;

		//Neon�汾��˹��Ԫ
		time = 0;
		generate_gauss();
		gettimeofday(&start, NULL);
		Gauss_SIMD_Neon();
		gettimeofday(&end, NULL);
		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		cout << "Neon�汾��˹��Ԫ:" << time << "ms" << endl;

		cout << endl;
	}
}

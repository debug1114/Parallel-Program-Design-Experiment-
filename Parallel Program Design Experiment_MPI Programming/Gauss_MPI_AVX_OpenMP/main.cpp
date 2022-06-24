#include<iostream>
#include<fstream>
#include<cmath>
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>
#include<immintrin.h>
#include<omp.h>

#define PROGRESS_NUM 8
#define THREAD_NUM 8

const int n = 1000;
float m[n][n];

void show(int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			std::cout << m[i][j] << " ";
		std::cout << std::endl;
	}
}
/*
void* PT_Static_Div_Elem_AVX(void* param) {  // ����ѭ��ȫ������
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 va, vt, vaik, vakj, vaij, vx;
	int j = 0;
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
		if (t_id == 0)
		{
			vt = _mm256_set1_ps(mat[k][k]);  // �Գ����㷨����SIMD���л�
			t1 = mat[k][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&mat[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&mat[k][j], va);
			}
			for (j; j < n; j++)
			{
				mat[k][j] = mat[k][j] / t1;  // �ƺ�
			}
			mat[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Divsion[i]);
			}
		}

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			vaik = _mm256_set1_ps(mat[i][k]);
			t2 = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0;
		}
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
			}
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
			}
		}
		else {
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}*/

float** generate(int n) {
	int k = 0;
	std::ifstream inp("input.txt");
	inp >> k;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

int getEnd(int rank)
{
	int t_end;
	if (rank == PROGRESS_NUM - 1) t_end = n - 1;
	else t_end = (rank + 1) * (n / PROGRESS_NUM) - 1;
	return t_end;
}

int main(int argc, char* argv[])
{
	std::ifstream inp("input.txt");
	int pn;
	inp >> pn;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			inp >> m[i][j];

	//m1 = generate(n);  // ʹ�����������Ķ�̬��ʼ�����ܻ�������⣬һ��ʹ�þ�̬

	//show(n);  // չʾ��ʼ�����

	MPI_Init(0, 0);
	int rank;
	//MPI_Status status;
	double head, tail;
	head = MPI_Wtime();

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// �黮��
	int t_begin = rank * (n / PROGRESS_NUM);
	int t_end = getEnd(rank);

	if (rank == 0) {
		// �ֹ�
		int tasknum, t1, t2;
		for (int j = 1; j < PROGRESS_NUM; j++)
		{
			t1 = j * (n / PROGRESS_NUM);
			t2 = getEnd(j);
			tasknum = n * (t2 - t1 + 1);
			MPI_Send(&m[t1][0], tasknum, MPI_FLOAT, j, n + 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&m[t_begin][0], n * (t_end - t_begin + 1), MPI_FLOAT, 0, n + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	float d_temp, e_temp;  // ʹ��d_temp������temp����e_temp����ȥtemp����������Ծ���ڴ���ʴ���
	int i, k, j = 0;
	__m256 va, vt, vaik, vakj, vaij, vx;
#pragma omp parallel if(1), num_threads(THREAD_NUM), private(i, j, k, va, vt, vaik, vakj, vaij, vx, d_temp, e_temp)
	for (int k = 0; k < n; k++)
	{
		if (rank == 0)
		{	//0�Ž��̸������
			vt = _mm256_set1_ps(m[k][k]);  // �Գ����㷨����SIMD���л�
			d_temp = m[k][k];

			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&m[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&m[k][j], va);
			}
			for (j; j < n; j++)
			{
				m[k][j] = m[k][j] / d_temp;  // �ƺ�
			}
			m[k][k] = 1.0;
			for (int j = 1; j < PROGRESS_NUM; j++)
			{
				MPI_Send(&m[k][0], n, MPI_FLOAT, j, k + 1, MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&m[k][0], n, MPI_FLOAT, 0, k + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (t_end > k)
		{
			for (int i = std::max(k + 1, t_begin); i <= t_end; i++)
			{  // ��������黮��˼��
				vaik = _mm256_set1_ps(m[i][k]);
				e_temp = m[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&m[k][j]);
					vaij = _mm256_loadu_ps(&m[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&m[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					m[i][j] -= e_temp * m[k][j];
				}
				m[i][k] = 0;
				if (i == k + 1 && rank)
				{
					MPI_Send(&m[i][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
		}
		if (!rank && t_end < k + 1 && n > k + 1)
		{
			MPI_Recv(&m[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tail = MPI_Wtime();

	MPI_Finalize();
	if (!rank)
	{
		std::cout << "MPI_GE_Cycle��" << (tail - head) * 1000 << " ms" << std::endl;
		//show(n);
	}
	return 0;

}

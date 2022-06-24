#include<iostream>
#include<fstream>
#include<cstring>
#include<omp.h>
#include<time.h>
#include"mpi.h"
#include <windows.h>
#include <immintrin.h>
#define  NUM_THREADS 4//线程数量
using namespace std;
//特殊高斯
float** a;
float** b;
int sum;//消元子个数
int n1, n2, l;
struct threadParam_t
{
	int t_id;
};
void inital(int n1, int n2, int l)
{
	a = new float* [n1];
	for (int i = 0; i < n1; i++)
		a[i] = new float[l];
	b = new float* [l];//首项为i的消元子为b[i]
	sum = n2;
	for (int i = 0; i < l; i++)//消元子
		b[i] = new float[l];
}
int lp(float** t, int temp, int l)
{
	int i;
	for (i = 0; i < l; i++)
	{
		if (t[temp][i] == 1.0)
			return i;
	}
	return i;
}
void Method_speG(int n1, int n2, int l)
{
	for (int i = 0; i < n1; i++)
	{
		while (lp(a, i, l) != l)
		{
			int temp = lp(a, i, l);
			if (lp(b, temp, l) != l)
			{
#pragma omp parallel for num_threads(NUM_THREADS)
				for (int j = 0; j < l; j++)
				{
					if (a[i][j] == b[temp][j])//异或
						a[i][j] = 0;
					else
						a[i][j] = 1;
				}
			}
			else
			{
#pragma omp parallel for num_threads(NUM_THREADS)
				for (int j = 0; j < l; j++)
				{
					b[temp][j] = a[i][j];
					//if (a[i][j] == 1)//用于检验正确性
					//	cout << 129 - j << " ";
					//sum = lp(a, i, l);
				}
				/*cout << endl;*/
			}
		}
	}
}
int transfer(string temp)
{
	int r = 0;
	for (int i = 0; i < temp.length(); i++)
	{
		r = r * 10 + temp[i] - '0';
	}
	return r;
}
int main(int argc, char* argv[])
{
	n1 = 53;
	n2 = 170;
	l = 562;
	inital(n1, n2, l);
	//------------------------------矩阵初始化------------------------//
	for (int i = 0; i < n1; i++)
	{
		for (int j = 0; j < l; j++)
			a[i][j] = 0;
	}
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
			b[i][j] = 0;
	}
	//-------------------------消元子读入----------------------------//
	ifstream infile("1_3.txt");//消元子
	int x = 0, y = 0;
	string temp;
	string tempforwards;
	infile >> temp;
	tempforwards = "562";
	x = 561 - transfer(temp);
	while (!infile.eof())
	{
		y = l - transfer(temp) - 1;
		b[x][y] = 1;
		if (transfer(tempforwards) > transfer(temp))
		{
			tempforwards = temp;
			infile >> temp;
			if (transfer(tempforwards) <= transfer(temp))
			{
				x = 561 - transfer(temp);
				tempforwards = "562";
			}
		}
	}
	ifstream infile2("2_3.txt");//消元子
	x = 0, y = 0;
	infile2 >> temp;
	tempforwards = "562";
	while (!infile2.eof())
	{
		y = l - transfer(temp) - 1;
		a[x][y] = 1;
		if (transfer(tempforwards) > transfer(temp))
		{
			tempforwards = temp;
			infile2 >> temp;
			if (transfer(tempforwards) <= transfer(temp))
			{
				x++;
				tempforwards = "562";
			}
		}
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//Method_speG(n1, n2, l);

	//----------------------------------------特殊高斯消去------------------------------------//
	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//获取id
	MPI_Comm_size(MPI_COMM_WORLD, &size);//获取数量
	int tag = 5;
	//划分任务
	for (int i = 0; i < n1; i++)
	{
		if (rank * (n1 / size) <= i && i < rank * (n1 / size) + n1 / size)
		{
			__m128 v0;
			while (lp(a, i, l) != l)
			{
				int temp = lp(a, i, l);
				if (lp(b, temp, l) != l)
				{
					//OpenMP
#pragma omp parallel for num_threads(NUM_THREADS)
					for (int j = 0; j < l; j++)
					{
						if (a[i][j] == b[temp][j])//异或
							a[i][j] = 0;
						else
							a[i][j] = 1;
					}
					//发送消息
					for (int j = 0; j < size; j++)
					{
						if(j!=rank)
							MPI_Send(&a[i][0], l, MPI_FLOAT, j, tag, MPI_COMM_WORLD);
					}
				}
				else
				{
					//OpenMP
#pragma omp parallel for num_threads(NUM_THREADS)
					for (int j = 0; j < l; j++)
					{
						//SIMD
						v0 = _mm_load_ps(&a[i][j]);
						_mm_store_ps(&b[temp][j], v0);

					}
					//发送消息
					for (int j = 0; j < size; j++)
					{
						//由于无法定位，将b[l][l]全部发送
						if (j != rank)
							for (int i = 0; i < l; i++)
							{
								MPI_Send(&b[i][0], l , MPI_FLOAT, j, tag + 1, MPI_COMM_WORLD);
							}
					}
				}
			}
		}
		else
		{
			//接受消息
			MPI_Recv(&a[i][0], l, MPI_FLOAT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
			for (int i = 0; i < l; i++);
				MPI_Recv(&b[i][0], l, MPI_FLOAT, MPI_ANY_SOURCE, tag + 1, MPI_COMM_WORLD, &status);
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << (tail - head) * 1000.0 / freq << "ms" << endl;
	MPI_Finalize();
}

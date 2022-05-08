#ifndef PREPTHREAD_H_INCLUDED
#define PREPTHREAD_H_INCLUDED

#pragma once
#define HAVE_STRUCT_TIMESPEC
#pragma comment(lib, "pthreadVC2.lib")
#include <pthread.h>
#include <semaphore.h>

using namespace std;

//线程控制变量
typedef struct
{
	int t_id; //线程编号
} threadParam_t;  //传参打包


//pthread2_sem信号量定义
const int THREAD_NUM = 4; //线程数
sem_t sem_main;
sem_t sem_workerstart[THREAD_NUM]; // 每个线程有自己专属的信号量
sem_t sem_workerend[THREAD_NUM];

//pthread3_sem信号量定义
sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];

//barrier 定义
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;


//pthread - 动态线程
typedef struct
{
	int k; //消去的轮次
	int t_id; // 线程 id
}threadParam_t_dynamic;


//全局变量
int N;
int Step[3] = { 1024,4096.8192 };
//设置地址对齐策略
__declspec(align(16))float** Gauss;//待消元的矩阵
__declspec(align(16))float** UP;//上三角矩阵
const int L = 100;
const int LOOP = 1;



//处理函数

//N阶上三角阵（每个元素均为1）
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

//生成待高斯消元的矩阵
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

//判断
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

//打印
void print(float** A, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			cout << A[i][j] << ",";
		cout << endl;
	}
}



//高斯消元-SIMD

//串行算法:普通高斯消元
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

//SSE指令高斯消元
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

//AVX指令高斯消元
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

//线程函数定义

void* threadFunc2_sem(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

void* threadFunc3_sem(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		// 比信号量更简洁的同步方式是使用 barrier
		if (t_id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}

		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}
			for (int i = 0; i < THREAD_NUM - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}

	pthread_exit(NULL);
	return NULL;
}

void* threadFunc3_barrier(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		if (t_id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}

		//第一个同步点
		pthread_barrier_wait(&barrier_Division);

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}

		// 第二个同步点
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(NULL);
	return NULL;
}

//动态线程
void* threadFunc_dynamic(void* param)
{
	threadParam_t_dynamic* p = (threadParam_t_dynamic*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
	{
		for (int j = k + 1; j < N; ++j)
		{
			Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
		}
		Gauss[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}

//REMIX
void* threadFunc2_sem_SSE(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			__m128 vaik = _mm_set1_ps(Gauss[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				__m128 vakj = _mm_loadu_ps(Gauss[k] + j);
				__m128 vaij = _mm_loadu_ps(Gauss[i] + j);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(Gauss[i] + j, vaij);
			}
			for (j; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[k][j] * Gauss[i][k];
			}
			Gauss[i][k] = 0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

void* threadFunc2_sem_AVX(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			__m256 vaik = _mm256_set1_ps(Gauss[i][k]);
			int j = k + 1;
			for (j; j + 8 <= N; j = j + 8)
			{
				__m256 vakj = _mm256_loadu_ps(Gauss[k] + j);
				__m256 vaij = _mm256_loadu_ps(Gauss[i] + j);
				__m256 vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(Gauss[i] + j, vaij);
			}
			for (j; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[k][j] * Gauss[i][k];
			}
			Gauss[i][k] = 0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

//高斯消元-pthread

// pthread并行 信号量 二重循环
void Gauss_pthread2_sem() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	threadParam_t param[THREAD_NUM];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2_sem, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}

}

// pthread并行 信号量 三重循环
void Gauss_pthread3_sem()
{
	//初始化信号量
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < THREAD_NUM - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

	//创建线程
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	threadParam_t param[THREAD_NUM];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc3_sem, (void*)(&param[t_id]));
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	// 销毁所有信号量
	sem_destroy(&sem_leader);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
}

void Gauss_pthread3_barrier() {
	//初始化 barrier
	pthread_barrier_init(&barrier_Division, NULL, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, NULL, THREAD_NUM);
	//创建线程
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	threadParam_t param[THREAD_NUM];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc3_barrier, (void*)(&param[t_id]));
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}

//动态
void Gauss_pthread_dynamic()
{
	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//创建工作线程，进行消去操作

		pthread_t handles[THREAD_NUM];// 创建对应的 Handle
		threadParam_t_dynamic param[THREAD_NUM];// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			pthread_create(&handles[t_id], NULL, threadFunc_dynamic, (void*)(&param[t_id]));
		}
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}



//REMIX

// pthread并行 信号量 二重循环 SSE向量化
void Gauss_pthread2_sem_SSE() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	threadParam_t param[THREAD_NUM];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2_sem_SSE, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做SSE除法操作
		__m128 vt = _mm_set1_ps(Gauss[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			__m128 va = _mm_loadu_ps(Gauss[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//开始唤醒工作线程
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}

}

// pthread并行 信号量 二重循环 AVX向量化
void Gauss_pthread2_sem_AVX() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	threadParam_t param[THREAD_NUM];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2_sem_AVX, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做AVX除法操作
		__m256 vt = _mm256_set1_ps(Gauss[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 8 <= N; j = j + 8)
		{
			__m256 va = _mm256_loadu_ps(Gauss[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//开始唤醒工作线程
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}

}

#endif // PREPTHREAD_H_INCLUDED

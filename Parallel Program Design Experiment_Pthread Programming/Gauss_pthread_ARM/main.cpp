#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
using namespace std;


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

typedef struct
{
	int k; //消去的轮次
	int t_id; // 线程 id
}threadParam_t_dynamic;

int N;
int Step[1] = { 500 };
//设置地址对齐策略
float** Gauss;//待消元的矩阵
float** UP;//上三角矩阵
const int L = 100;
const int LOOP = 1;

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

// neon并行算法
void Gauss_neon()
{
	for (int k = 0; k < N; k++)
	{
		float32x4_t Akk = vmovq_n_f32(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			float32x4_t Akj = vld1q_f32(Gauss[k] + j);
			Akj = vdivq_f32(Akj, Akk);
			vst1q_f32(Gauss[k] + j, Akj);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			float32x4_t Aik = vmovq_n_f32(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				float32x4_t Akj = vld1q_f32(Gauss[k] + j);
				float32x4_t Aij = vld1q_f32(Gauss[i] + j);
				float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
				Aij = vsubq_f32(Aij, AikMulAkj);
				vst1q_f32(Gauss[i] + j, Aij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
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

void* threadFunc2_sem_neon(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			float32x4_t vaik = vmovq_n_f32(Gauss[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				float32x4_t vakj = vld1q_f32(Gauss[k] + j);
				float32x4_t vaij = vld1q_f32(Gauss[i] + j);
				float32x4_t vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(Gauss[i] + j, vaij);
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

// pthread并行 信号量 二重循环 NEON向量化
void Gauss_pthread2_sem_neon() {
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
		pthread_create(&handles[t_id], NULL, threadFunc2_sem_neon, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做NEON除法操作
		float32x4_t vt = vmovq_n_f32(Gauss[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			float32x4_t va = vld1q_f32(Gauss[k] + j);
			va = vdivq_f32(va, vt);
			vst1q_f32(Gauss[k] + j, va);
		}
		for (j; j < N; j++)
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

int main()
{
	struct timeval start;
	struct timeval end;
	float time = 0;
	for (int j = 0; j < 1; j++)
	{

		generate_up(Step[j]);
		cout << "N=" << N << endl;

		//普通高斯消元：串行算法
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			//print_matrix();
			gettimeofday(&start, NULL);
			Gauss_ordinary();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "serial:" << time / LOOP << "ms" << endl;
		//print_matrix();

		//SIMD高斯消元：NEON
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_neon();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "neon:" << time / LOOP << "ms" << endl;
		//print_matrix();

		// pthread-两重 信号量
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_pthread2_sem();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread2_sem:" << time / LOOP << "ms" << endl;
		//print_matrix();


		// pthread-三重 信号量
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_pthread3_sem();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread3_sem:" << time / LOOP << "ms" << endl;
		//print_matrix();

		//  pthread-三重 信号量 barrier
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_pthread3_barrier();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread3_barrier:" << time / LOOP << "ms" << endl;
		//print_matrix();

		//  pthread-动态线程
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_pthread_dynamic();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread_dynamic:" << time / LOOP << "ms" << endl;
		//print_matrix();

		//  pthread-两重 信号量 NEON
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			generate_gauss();
			gettimeofday(&start, NULL);
			Gauss_pthread2_sem_neon();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread2_sem_neon:" << time / LOOP << "ms" << endl;
		//print_matrix();
	}

}



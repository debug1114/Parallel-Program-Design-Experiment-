#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
using namespace std;


typedef struct
{
	int t_id; //�̱߳��
} threadParam_t;  //���δ��


//pthread2_sem�ź�������
const int THREAD_NUM = 4; //�߳���
sem_t sem_main;
sem_t sem_workerstart[THREAD_NUM]; // ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend[THREAD_NUM];

//pthread3_sem�ź�������
sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];

//barrier ����
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct
{
	int k; //��ȥ���ִ�
	int t_id; // �߳� id
}threadParam_t_dynamic;

int N;
int Step[1] = { 500 };
//���õ�ַ�������
float** Gauss;//����Ԫ�ľ���
float** UP;//�����Ǿ���
const int L = 100;
const int LOOP = 1;

//��˹��Ԫ-SIMD

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

// neon�����㷨
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


//�̺߳�������
void* threadFunc2_sem(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����

		//ѭ����������
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//��ȥ
			for (int j = k + 1; j < N; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
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
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
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
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//��ȥ
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
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
			}
			for (int i = 0; i < THREAD_NUM - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
			}
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
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
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		if (t_id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}

		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Division);

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < N; i += THREAD_NUM)
		{
			//��ȥ
			for (int j = k + 1; j < N; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}

		// �ڶ���ͬ����
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
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����

		//ѭ����������
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
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return NULL;
}

//��̬�߳�
void* threadFunc_dynamic(void* param)
{
	threadParam_t_dynamic* p = (threadParam_t_dynamic*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
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

// pthread���� �ź��� ����ѭ��
void Gauss_pthread2_sem() {
	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	threadParam_t param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2_sem, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//���߳�����������
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����

		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//���������ź���
	sem_destroy(&sem_main);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}

}

// pthread���� �ź��� ����ѭ��
void Gauss_pthread3_sem()
{
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < THREAD_NUM - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	threadParam_t param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc3_sem, (void*)(&param[t_id]));
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	// ���������ź���
	sem_destroy(&sem_leader);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
}

void Gauss_pthread3_barrier() {
	//��ʼ�� barrier
	pthread_barrier_init(&barrier_Division, NULL, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, NULL, THREAD_NUM);
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	threadParam_t param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc3_barrier, (void*)(&param[t_id]));
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//�������е� barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}

//��̬
void Gauss_pthread_dynamic()
{
	for (int k = 0; k < N; ++k)
	{
		//���߳�����������
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//���������̣߳�������ȥ����

		pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
		threadParam_t_dynamic param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
		//��������
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			pthread_create(&handles[t_id], NULL, threadFunc_dynamic, (void*)(&param[t_id]));
		}
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}

// pthread���� �ź��� ����ѭ�� NEON������
void Gauss_pthread2_sem_neon() {
	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	threadParam_t param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2_sem_neon, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//���߳���NEON��������
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
		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����

		for (int t_id = 0; t_id < THREAD_NUM; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//���������ź���
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

		//��ͨ��˹��Ԫ�������㷨
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

		//SIMD��˹��Ԫ��NEON
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

		// pthread-���� �ź���
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


		// pthread-���� �ź���
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

		//  pthread-���� �ź��� barrier
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

		//  pthread-��̬�߳�
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

		//  pthread-���� �ź��� NEON
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



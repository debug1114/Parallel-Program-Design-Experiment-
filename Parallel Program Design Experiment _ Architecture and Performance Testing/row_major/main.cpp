#include <iostream>

using namespace std;

const int N = 100; // matrix size

double b[N][N], col_sum[N],a[N];

void init(int n) // generate a N*N matrix
{
	 for (int i = 0; i < N; i++)
     {
         a[i] = i;
         for(int j = 0; j < N; j++)
            b[i][j] = i + j;
     }
}

int main()
{
	init(N);

	for (int i = 0; i < N; i++)
		col_sum[i] = 0.0;
    for (int j = 0; j < N; j++)
    {
        for(int i = 0; i < N; i++)
            col_sum[i] += b[j][i] * a[j];
    }

 }

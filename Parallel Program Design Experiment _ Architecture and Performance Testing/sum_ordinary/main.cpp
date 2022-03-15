#include <iostream>

using namespace std;

const int n = 100;

int a[n], sum1, sum2, sum;

void init(int N)
{
    for(int i = 0; i < N; i++)
        a[i] = i;
}

int main()
{
    init(n);
    for(int i = 0 ;i < n ;i ++)
        sum += a[i];
}

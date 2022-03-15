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

    sum1 = 0;
    sum2 = 0;

    for(int i = 0; i < n; i += 2){
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;

}

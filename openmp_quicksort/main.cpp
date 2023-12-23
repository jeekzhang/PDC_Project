#include <iostream>
#include <vector>
#include <ctime>
#include<omp.h>
#include<algorithm>
#include<math.h>
using namespace std;

vector<int> randomIntList(int start, int stop, int length) {
	vector<int> randomList;
	for (int i = 0; i < length; ++i) {
		randomList.push_back(rand() % (stop - start + 1) + start);
	}
	return randomList;
}

int partition(vector<int>& arr, int start, int end) {
	int select = rand() % (end - start + 1) + start;
	swap(arr[select], arr[start]);
	int left = start, right = end;
	int tmp = arr[left];
	while (left < right) {
		while (left < right && arr[right] >= tmp) {
			right--;
		}
		arr[left] = arr[right];
		while (left < right && arr[left] < tmp) {
			left++;
		}
		arr[right] = arr[left];
	}
	arr[left] = tmp;
	return left;
}

void quickSort(vector<int>& arr, int start, int end) {
	if (start < end) {
		int p = partition(arr, start, end);
		quickSort(arr, start, p - 1);
		quickSort(arr, p + 1, end);
	}
}

void parallel_quickSort(vector<int>& arr, int start, int end) {
	if (start < end) {
		int p = partition(arr, start, end);
		#pragma omp parallel sections
		{
			#pragma omp section
			parallel_quickSort(arr, start, p - 1);
			#pragma omp section
			parallel_quickSort(arr, p + 1, end);
		}
	}
}

void _parallel_quickSort(vector<int>& arr, int start, int end, int v) {
	if (end - start < v) quickSort(arr, start, end); 
	else if (start < end) {
		int p = partition(arr, start, end);
		#pragma omp parallel sections
		{
			#pragma omp section
			_parallel_quickSort(arr, start, p - 1, v);
			#pragma omp section
			_parallel_quickSort(arr, p + 1, end, v);
		}
	}
}

void comp_eaxm() {
	int epoch = 1000;
	int m = 10000;
	int threads = 4;
	bool exam = true;
	double t1, t2, t3;
	t1 = t2 = t3 = 0;
	for (int i = 0; i < epoch; i++) {
		int len = rand() % m + m / 10;
		vector<int> arr = randomIntList(1, m, len);
		vector<int> prl_arr = arr;
		vector<int> serial_arr = arr;
		double start_t1 = omp_get_wtime();
		sort(arr.begin(), arr.end());
		double end_t1 = omp_get_wtime();
		t1 += end_t1 - start_t1;
		double start_t2 = omp_get_wtime();
		quickSort(serial_arr, 0, len-1);
		double end_t2 = omp_get_wtime();
		t2 += end_t2 - start_t2;
		double start_t3 = omp_get_wtime();
		omp_set_num_threads(threads);
		parallel_quickSort(prl_arr, 0, len-1);
		double end_t3 = omp_get_wtime();
		t3 += end_t3 - start_t3;
		bool arraysEqual = std::equal(arr.begin(), arr.end(), serial_arr.begin()) &&
		                       std::equal(arr.begin(), arr.end(), prl_arr.begin());
		exam &= arraysEqual;
	}
	cout << t1 << "s" << endl;
	cout << t2 << "s" << endl;
	cout << t3 << "s" << endl;
	if(exam)cout<<"AC!" << endl; else cout<<"WA!" << endl;
}
void findV() {
	for (int j=1;j<=4;j++) {
		int v=pow(10,j);
		int epoch = 1000;
		int m = 1000000;
		int threads = rand() % 7 + 2;
		double t1, t2, t3;
		t1 = t2 = t3 = 0;
		for (int i = 0; i < epoch; i++) {
			int len = rand() % 990001 + 10000;
			vector<int> arr = randomIntList(1, m, len);
			vector<int> prl_arr = arr;
			vector<int> serial_arr = arr;
			double start_t1 = omp_get_wtime();
			omp_set_num_threads(threads);
			_parallel_quickSort(arr, 0, len-1, v);
			double end_t1 = omp_get_wtime();
			t1 += end_t1 - start_t1;
			double start_t2 = omp_get_wtime();
			quickSort(serial_arr, 0, len-1);
			double end_t2 = omp_get_wtime();
			t2 += end_t2 - start_t2;
			double start_t3 = omp_get_wtime();
			omp_set_num_threads(threads);
			parallel_quickSort(prl_arr, 0, len-1);
			double end_t3 = omp_get_wtime();
			t3 += end_t3 - start_t3;
		}
		cout << v << ": " << endl;
		cout << t1 << "s" << endl;
		cout << t2 << "s" << endl;
		cout << t3 << "s" << endl;
	}
}
void speedup() {
	int epoch = 100000;
	int m = 10000000;
	int len[5]= {
		5000,10000,100000,1000000,10000000
	};
	for (int threadi=0; threadi<=5; threadi++) {
		int threads = pow(2,threadi);
		for (int i=0; i<5; i++) {
			vector<int> arr = randomIntList(1, m, len[i]);
			vector<int> s_arr = arr;
			double start_t = omp_get_wtime();
			omp_set_num_threads(threads);
			parallel_quickSort(arr, 0, len[i]-1);
			// _parallel_quickSort(arr, 0, len[i]-1, 1000);
			double end_t = omp_get_wtime();
			double t_p = end_t - start_t;
			start_t = omp_get_wtime();
			quickSort(s_arr, 0, len[i]-1);
			end_t = omp_get_wtime();
			double t_1 = end_t - start_t;
			double speedup_v = t_1/t_p;
			cout<<"threads: "<<threads<<" length: "<<len[i]<<" speedup: "<<speedup_v<<endl;
		}
	}
}
int main() {
	srand(3407);
	comp_eaxm();
	findV();
	speedup();
	return 0;
}
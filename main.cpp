#include <iostream>
#include <cstring>
#include <random>
#include <ctime>
#include <vector>
#include <thread>
#include <cstdint>

#include "omp.h"
// #include <boost/thread/barrier.hpp"
#include "BoostBarrier.h"
#include "BarrierEditDistance.h"

using namespace std;

int minimum(int a, int b, int c) {
	int min = a;

	if (b < min)
		min = b;
	if (c < min)
		min = c;

	return min;
}

int sequential_edit_distance(const char *A, const char *B) {
	const int M = strlen(A);
	const int N = strlen(B);

	uint16_t **D = new uint16_t*[M + 1];

	for (uint16_t i = 0; i < M + 1; i++)
		D[i] = new uint16_t[N + 1];

	for (uint16_t i = 0; i < M + 1; i++)
		D[i][0] = i;

	for (uint16_t j = 1; j < N + 1; j++)
		D[0][j] = j;

	for (int i = 1; i < M + 1; i++) {
		for (int j = 1; j < N + 1; j++) {
			if (A[i - 1] != B[j - 1]) {
				uint16_t k = minimum(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1]);
				D[i][j] = k + 1;
			} else {
				D[i][j] = D[i - 1][j - 1];
			}
		}
	}

	uint16_t distance = D[M][N];

	for (int i = 0; i < M + 1; i++)
		delete[] D[i];

	delete[] D;

	return distance;
}

void computeTile(const char *A, const char *B, const int M, const int N, uint16_t *D, uint16_t i, uint16_t j, const uint16_t TILE_SIZE) {
	uint16_t rowIndex = i * TILE_SIZE + 1;
	uint16_t colIndex = j * TILE_SIZE + 1;
	for (uint16_t i = rowIndex; i < N && i < rowIndex + TILE_SIZE; i++) {
		for (uint16_t j = colIndex; j < M && j < colIndex + TILE_SIZE; j++) {
			if (A[i - 1] != B[j - 1]) {
				uint16_t k = minimum(D[i * M + j - 1], D[(i - 1) * M + j],
						D[(i - 1) * M + j - 1]);
				D[i * M + j] = k + 1;
			} else {
				D[i * M + j] = D[(i - 1) * M + j - 1];
			}
		}
	}
}

int OMP_edit_distance(const char *A, const char *B) {
	const int M = strlen(A);
	const int N = strlen(B);

	const int length = (M + 1) * (N + 1);
	uint16_t *D = new uint16_t[length];

	for (uint16_t i = 0; i < M + 1; i++)
		D[i] = i;

	for (uint16_t j = 1; j < N + 1; j++)
		D[j * (M + 1)] = j;

	const uint16_t TILE_SIZE = 256;

	int M_Tiles = ceil((double) M / TILE_SIZE);
	int N_Tiles = ceil((double) N / TILE_SIZE);

	#pragma omp parallel num_threads(4)
	{
		#pragma omp master
		{
			for (int d = -M_Tiles + 1; d < N_Tiles; d++) {
				const int maxIndex = min(M_Tiles + d, N_Tiles);
				const int minIndex = max(d, 0);
				for (uint16_t i = minIndex; i < maxIndex; i++) {
					#pragma omp task
					{
						uint16_t j = M_Tiles + d - 1 - i;
						computeTile(A, B, M + 1, N + 1, &D[0], i, j, TILE_SIZE);
					}
			    }
            #pragma omp taskwait
			}
		}
	}

	uint16_t distance = D[N * (M + 1) + M];

	delete[] D;

	return distance;
}

int cpp_threads_edit_distance(const char *A, const char *B) {
	const int M = strlen(A);
	const int N = strlen(B);

	int length = (M + 1) * (N + 1);
	uint16_t *D = new uint16_t[length];

	for (uint16_t i = 0; i < M + 1; i++)
		D[i] = i;

	for (uint16_t j = 1; j < N + 1; j++)
		D[j * (M + 1)] = j;

	barrier resultBarrier(BarrierEditDistance::MAX_NUM_THREAD + 1);
	vector<thread> threadVector(BarrierEditDistance::MAX_NUM_THREAD);

	for (uint16_t i = 0; i < BarrierEditDistance::MAX_NUM_THREAD; i++) {
		threadVector[i] = thread(BarrierEditDistance(A, B, D, &resultBarrier));
		threadVector[i].detach();
	}

	resultBarrier.count_down_and_wait();
	BarrierEditDistance::numThreads = 0;

	uint16_t distance = D[N * (M + 1) + M];

	delete[] D;

	return distance;
}

int main() {

	const int M = 10000;
	const int N = 10000;

	char *A = new char[M];
	char *B = new char[N];

	char ASCII_letter_code;

	for (int i = 0; i < M; i++) {
		ASCII_letter_code = rand() % 2 + 65;
		A[i] = static_cast<char>(ASCII_letter_code);
	}

	for (int i = 0; i < N; i++) {
		ASCII_letter_code = rand() % 2 + 65;
		B[i] = static_cast<char>(ASCII_letter_code);
	}

	int distance;
	double start, end;
	double seqTime = 0, CPPTTime = 0, OMPTime = 0;
	int iterations = 1;

	for (int i = 0; i < iterations; i++) {
		cout << "Iteration no. " << i + 1 << endl;

		// Sequential version
		start = omp_get_wtime();
		distance = sequential_edit_distance(A, B);
		end = omp_get_wtime();
		seqTime += end - start;
		cout << "Sequential distance: " << distance << endl;

		// Tiled OpenMP version
		start = omp_get_wtime();
		distance = OMP_edit_distance(A, B);
		end = omp_get_wtime();
		OMPTime += end - start;
		cout << "Tiled OpenMP distance: " << distance << endl;

		// C++ threads version
		start = omp_get_wtime();
		distance = cpp_threads_edit_distance(A, B);
		end = omp_get_wtime();
		CPPTTime += end - start;
		cout << "C++ threads distance: " << distance << endl;

		cout << endl;

	}

	cout << endl;

	cout << "Average sequential execution time: " << seqTime / iterations
			<< " s" << endl;

	cout << "Average tiled OpenMP execution time: " << OMPTime / iterations
			<< " s" << endl;
	cout << "Average C++ threads execution time: " << CPPTTime / iterations
			<< " s" << endl;

	cout << endl;

	cout << "Tiled OpenMP speed-up: " << seqTime / OMPTime << endl;
	cout << "C++ Threads SpeedUp: " << seqTime / CPPTTime << endl;

	return 0;
}

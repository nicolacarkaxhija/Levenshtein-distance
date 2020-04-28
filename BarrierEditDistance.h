#include <cstring>
#include <cmath>
// #include <boost/thread/barrier.hpp>
#include "BoostBarrier.h"

using namespace std;

class BarrierEditDistance {

public:

	BarrierEditDistance(const char *A, const char *B, uint16_t *D, barrier *resultBarrier) : D(D), A(A), B(B), resultBarrier(resultBarrier) {
		M = strlen(A);
		N = strlen(B);
		TILES_PER_ROW = ceil((double) M / TILE_SIZE);
		TILES_PER_COL = ceil((double) N / TILE_SIZE);
		tid = numThreads++;
	}

	void operator()() {

		for (int d = -TILES_PER_ROW + 1; d < TILES_PER_COL; d++) {
			const int maxIndex = min(TILES_PER_ROW + d, TILES_PER_COL);
			const int minIndex = max(d, 0);

			int diagLength = maxIndex - minIndex;
			for (int k = 0; k * MAX_NUM_THREAD + tid < diagLength; k++) {
				uint16_t i = k * MAX_NUM_THREAD + tid + minIndex;
				uint16_t j = TILES_PER_ROW + d - i - 1;
				uint16_t tileRow = i * TILE_SIZE + 1;
				uint16_t tileCol = j * TILE_SIZE + 1;

				computeTile(tileRow, tileCol);
			}

			diagonalBarrier.count_down_and_wait();
		}
		resultBarrier->count_down_and_wait();
	}

	static const uint16_t TILE_SIZE = 256;
	static const uint16_t MAX_NUM_THREAD = 4;
	static barrier diagonalBarrier;
	static int numThreads;

private:

	int minimum(const int a, const int b, const int c) {
		int min = a;

		if (b < min)
			min = b;
		if (c < min)
			min = c;

		return min;
	}

	void computeTile(uint16_t tileRow, uint16_t tileCol) {
		uint16_t rows = M + 1;
		uint16_t cols = N + 1;
		for (uint16_t i = tileRow; i < cols && i < tileRow + TILE_SIZE; i++) {
			for (uint16_t j = tileCol; j < rows && j < tileCol + TILE_SIZE;
					j++) {
				if (A[i - 1] != B[j - 1]) {
					uint16_t k = minimum(D[i * rows + j - 1],
							D[(i - 1) * rows + j], D[(i - 1) * rows + j - 1]);
					D[i * rows + j] = k + 1;
				} else {
					D[i * rows + j] = D[(i - 1) * rows + j - 1];
				}
			}
		}
	}

	uint16_t *D;
	const char *A, *B;
	int M, N, TILES_PER_ROW, TILES_PER_COL;
	barrier *resultBarrier;
	uint16_t tid;
};

#ifndef	DATA_H
#define	DATA_H

#include	<stdio.h>
#include	<stdlib.h>
#include	"../../tt.h"

void load(TTTensor** x, TTTensor** y, const char* imagePath, const char* labelPath, int n) {
	FILE* f = fopen(imagePath, "rb");
	float* dataX = (float*)malloc(sizeof(float) * n * 784);
	float* dataY = (float*)calloc(n * 10, sizeof(float));
	unsigned char byte;
	
	fseek(f, 16, SEEK_SET);
	
	for(int i = 0; i < n * 784; i++) {
		fread(&byte, 1, 1, f);
		dataX[i] = (float)byte / 128 - 1;
	}

	fclose(f);
	f = fopen(labelPath, "rb");
	fseek(f, 8, SEEK_SET);

	for(int i = 0; i < n; i++) {
		fread(&byte, 1, 1, f);
		dataY[i * 10 + byte] = 1;
	}

	*x = ttNewTensor(dataX, (int[]){n, 1, 28, 28}, 4);
	*y = ttNewTensor(dataY, (int[]){n, 10}, 2);
	free(dataX);
	free(dataY);
	fclose(f);
}

int main() {
	TTTensor* x;
	TTTensor* y;

	ttInitialize();
	load(&x, &y, "train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000);
	ttSave(x, "train_x.tt");
	ttSave(y, "train_y.tt");
	printf("Train set saved.\n");
	load(&x, &y, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);
	ttSave(x, "test_x.tt");
	ttSave(y, "test_y.tt");
	printf("Test set saved.\n");
	ttTerminate();
}

#endif
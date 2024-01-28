#include	<stdio.h>
#include	<stdlib.h>
#include	"cnn.h"
#include	"../../tt.h"
#include	"../../nn.h"

int main() {
	ttInitialize();
	
	TTTensor* x = ttLoad("test_x.tt");
	TTTensor* y = ttLoad("test_y.tt");

	for(int e = 0; e < 5; e++) {
		char prefix[10];

		sprintf(prefix, "e%d_", e + 1);
		printf("epoch %d\n", e + 1);

		CNNParams params = loadCNN("trained", prefix);
		int correct = 0;
		float avgLoss = 0;

		for(int b = 0; b < 10000; b++) {
			ttPushTensorAllocator();

			TTTensor* sample = ttGet(x, (TTIndexer[]){ttAt(b)}, 1);
			TTTensor* ground = ttGet(y, (TTIndexer[]){ttAt(b)}, 1);
			TTTensor* o = CNN(params, sample);
			TTTensor* loss = ttCrossEntropy(o, ground);

			int maxO = 0;

			for(int i = 0; i < 10; i++) {
				if(o->data[i] > o->data[maxO])
					maxO = i;
			}

			correct += y->data[10 * b + maxO] == 1;
			avgLoss += loss->data[0];
			ttPopTensorAllocator();
		}

		printf("epoch %d, acc: %.2f%%, avg loss: %.4f\n",
			e + 1, (float)correct / 10000 * 100, avgLoss / 10000
		);
	}

	ttTerminate();
	return 0;
}
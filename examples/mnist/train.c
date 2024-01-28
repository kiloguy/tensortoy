#include	<stdio.h>
#include	<stdlib.h>
#include	"cnn.h"
#include	"../../tt.h"
#include	"../../nn.h"

int main() {
	ttInitialize();
	
	TTTensor* x = ttLoad("train_x.tt");
	TTTensor* y = ttLoad("train_y.tt");;
	CNNParams params = newCNN();
	TTTensor* paramsTensors[] = {
		params.line1.W, params.line1.b, params.line2.W, params.line2.b,
		params.conv1.W, params.conv1.b, params.conv2.W, params.conv2.b
	}; // The tensors that need to optimize.

	for(int e = 0; e < 5; e++) {
		int correct = 0;

		for(int b = 0; b < 60000; b++) {
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

			bool c = y->data[10 * b + maxO] == 1;

			if(c)
				correct += 1;

			printf("epoch %d, sample %d, correct: %d, acc so far: %.2f%%, loss: %.4f\n",
				e + 1, b + 1, c, (float)correct / (b + 1) * 100, loss->data[0]
			);

			ttZeroGrad(paramsTensors, 8);
			ttBackward(loss);
			ttOptimize(paramsTensors, 8, 0.01);

			ttPopTensorAllocator();
		}

		char prefix[10];

		sprintf(prefix, "e%d_", e + 1);
		saveCNN(params, "trained", prefix);
	}

	ttTerminate();
	return 0;
}
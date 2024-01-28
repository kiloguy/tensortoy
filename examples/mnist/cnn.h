#ifndef	CNN_H
#define	CNN_H

#include	<stdio.h>
#include	"../../tt.h"
#include	"../../nn.h"

typedef struct {
	TTConv2DParams conv1;
	TTConv2DParams conv2;
	TTLinearParams line1;
	TTLinearParams line2;
} CNNParams;

CNNParams newCNN() {
	return (CNNParams) {
		.conv1 = ttNewConv2D(1, 4, 5),
		.conv2 = ttNewConv2D(4, 8, 5),
		.line1 = ttNewLinear(8 * 14 * 14, 64),
		.line2 = ttNewLinear(64, 10)
	};
}

TTTensor* CNN(CNNParams params, TTTensor* x) {
	TTTensor* o = ttConv2D(params.conv1, x);

	o = ttMaxPool2D(o, 4);
	o = ttConv2D(params.conv2, o);
	o = ttMaxPool2D(o, 4);
	o = ttReshape(o, (int[]){8 * 14 * 14}, 1);
	o = ttSigmoid(ttLinear(params.line1, o));
	o = ttSigmoid(ttLinear(params.line2, o));
	return o;
}

void saveCNN(CNNParams params, const char* dir, const char* prefix) {
	char path[32];

	sprintf(path, "%s/%sline1.W.tt", dir, prefix); ttSave(params.line1.W, path);
	sprintf(path, "%s/%sline1.b.tt", dir, prefix); ttSave(params.line1.b, path);
	sprintf(path, "%s/%sline2.W.tt", dir, prefix); ttSave(params.line2.W, path);
	sprintf(path, "%s/%sline2.b.tt", dir, prefix); ttSave(params.line2.b, path);
	sprintf(path, "%s/%sconv1.W.tt", dir, prefix); ttSave(params.conv1.W, path);
	sprintf(path, "%s/%sconv1.b.tt", dir, prefix); ttSave(params.conv1.b, path);
	sprintf(path, "%s/%sconv2.W.tt", dir, prefix); ttSave(params.conv2.W, path);
	sprintf(path, "%s/%sconv2.b.tt", dir, prefix); ttSave(params.conv2.b, path);
}

CNNParams loadCNN(const char* dir, const char* prefix) {
	char path[32];
	CNNParams params = newCNN();

	sprintf(path, "%s/%sline1.W.tt", dir, prefix); params.line1.W = ttLoad(path);
	sprintf(path, "%s/%sline1.b.tt", dir, prefix); params.line1.b = ttLoad(path);
	sprintf(path, "%s/%sline2.W.tt", dir, prefix); params.line2.W = ttLoad(path);
	sprintf(path, "%s/%sline2.b.tt", dir, prefix); params.line2.b = ttLoad(path);
	sprintf(path, "%s/%sconv1.W.tt", dir, prefix); params.conv1.W = ttLoad(path);
	sprintf(path, "%s/%sconv1.b.tt", dir, prefix); params.conv1.b = ttLoad(path);
	sprintf(path, "%s/%sconv2.W.tt", dir, prefix); params.conv2.W = ttLoad(path);
	sprintf(path, "%s/%sconv2.b.tt", dir, prefix); params.conv2.b = ttLoad(path);
	return params;
}

#endif
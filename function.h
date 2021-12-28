#pragma once

#ifndef __FUNCTION_H
#define __FUNCTION_H
#include <cmath>

double lostvalue(double y, double y_);
double sigmoid(double z);
double relu(double z); 
double d_relu(double z);


double lostvalue(double y, double y_) {
	double res = 0;
	res = (y - 1.0) * log(1.0 - y_) - y * log(y_);
	return res;
}
double sigmoid(double z){
	double res = 0;
	res = 1.0 / (1 + exp(-z));
	return res;
}
double relu(double z) { 
	return (z >= 0) ? z : 0.001 * z; 
}
double d_relu(double z) {
	return (z >= 0) ? 1 : 0.001;
}

#endif // __FUNCTION_H


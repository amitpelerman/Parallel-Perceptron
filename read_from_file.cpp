//Amit Pelerman
//203518006
#define _CRT_SECURE_NO_WARNINGS
#include "app.h"

double* readFromFile(int* N, int* K, double* a0, double* aMax, int* LIMIT, double* QC, char* fileName, int*& signArray) {
	FILE *file = fopen(fileName, "r");
	if (!file)
	{
		printf("ERROR via opening file\n");
		return 0;
	}

	fscanf(file, "%d", N);
	fscanf(file, "%d", K);
	double* input = new double[*N* (*K)];
	signArray = new int[*N];
	fscanf(file, "%lf", a0);
	fscanf(file, "%lf", aMax);
	fscanf(file, "%d", LIMIT);
	fscanf(file, "%lf", QC);
	{
		for (int i = 0; i < *N; i++)
		{
			for (int z = 0; z < (*K); z++)
			{
				if (feof(file))
					break;
				fscanf(file, "%lf", &input[(i*(*K)) + z]);
			}
			fscanf(file, "%d", &signArray[i]);
		}
	}
	return input;
	fclose(file);
}
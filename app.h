#ifndef app_h
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
using namespace std;
#pragma warning(disable:4996)

//cuda
__global__ void calc(double* array, int* results, double* weight, int N, int K);
cudaError_t calculateWithCuda(double* array, int* results, double* weight, int N, int K);

//Slave
void slaveProc(MPI_Status& status);
void receiveDataToWorkFromMaster(MPI_Status& status, double& a, double& q, int& N, int& K, int& LIMIT, double *&input, int *&signArray, double*& weight);
void slaveGotWork(MPI_Status& status, double& bias, int N, int K, double a, double q, int LIMIT, double* weight, double* input, int* signArray);
void sendResultsToMaster(double q, double a, double* weight, double bias, int K);
double calcSum(double* input, int i, int K, double* weight);
double calcError(int N, int K, double *input, int *signArray, double *weight, double &bias, double& a, int LIMIT);
int checkSign(double w);
int countNmis(int N, int K, double bias, double* input, int* signArray, double* weight);
void fixWeights(double *weight, int size, double& bias, double a, double* cordinates, int error);

//Master
void masterProc(MPI_Status& status, int numOfProcs);
bool overNumOfProcs(double a0, double aMax, int numOfProcs);
void sendFileDataToSlave(int N, int K, int LIMIT, double* input, int* signArray, int i);
void sendTasksToSlave(int K, double& a, double* weight, int i);
void terminateSlaves(double recvA, double& finalA, double& a, int slaveNum, double* finalWeight, double* weight, double& finalQ, double q, double& finalBias, double& bias, int K);
void receiveResultsFromSlave(double& q, MPI_Status& status, double& recvA, double* weight, double& bias, int K, int& source);
void initWeightsToZero(double* weightArray, int size);
char* createFilePath(char* folder, char* fileName);
void alphaIncrement(double& a, double a0);

//Prints
void printWeights(double* finalWeight, double finalBias, int K);
void printLinearEquation(double* weights, double bias, int K);
void printResults(double finalQ, double finalA, double* finalWeight, double finalBias, double startTime, double QC, int K);

//Files- read & write
void writeToFile(double finalA, double finalQ, double* finalWeight, double finalBias, double QC, double startTime, int K, char* fileName);
double* readFromFile(int* N, int* K, double* a0, double* aMax, int* LIMIT, double* QC, char* fileName, int* & signArray);

#define TERMINATE_TAG 1
#define WORK_TAG 0
#endif // !app_h


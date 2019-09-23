//Amit Pelerman
//203518006
#include "app.h"

char* FILE_SRC_NAME = "data1.txt";
char* FILE_DEST_NAME = "output.txt";
char* FOLDER = "C:\\";

int main(int argc, char *argv[])
{
	int myId, numOfProcs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Status status;

	if (myId == 0)	//myId 0 - Master
	{
		masterProc(status, numOfProcs);
	}
	else 	//Slave
	{
		slaveProc(status);
	}
	MPI_Finalize();
	return 0;
}

//Slave
void slaveProc(MPI_Status& status)
{
	double a, q, bias = 1, *input = nullptr, *weight = nullptr, *finalWeight = nullptr;
	int N, K, LIMIT, *signArray = nullptr;
	//Receive data by reff from the Master then work
	receiveDataToWorkFromMaster(status, a, q, N, K, LIMIT, input, signArray, weight);
	//Receive Datas and work until master send Terminate
	while (status.MPI_TAG == WORK_TAG)
	{
		slaveGotWork(status, bias, N, K, a, q, LIMIT, weight, input, signArray);
	}
}

//Master
void masterProc(MPI_Status& status, int numOfProcs)
{
	double recvA, finalA, q = 0, finalQ, finalBias = 1, bias = 1, QC, a0, aMax, a;
	int slaveNum = 0, N, K, LIMIT, *signArray = nullptr;
	double* input = nullptr, *weight = nullptr, *finalWeight = nullptr;
	char *filePathRead = createFilePath(FOLDER, FILE_SRC_NAME);
	double startTime = MPI_Wtime();

	input = readFromFile(&N, &K, &a0, &aMax, &LIMIT, &QC, filePathRead, signArray);
	finalA = aMax, finalQ = QC;
	if (overNumOfProcs(a0,aMax,numOfProcs))
	{
		cout << "Maximum processes for this input file is: " << (aMax/a0) + 1 << endl;
	}
	else
	{
		//Init Alpha
		a = a0;
		cout << "N: " << N << " K: " << K << " a0: " << a0 << " aMax: " << aMax << " LIMIT: " << LIMIT << " QC: " << QC << endl;

		//Init all weights to be 0
		weight = new double[K];
		initWeightsToZero(weight, K);
		finalWeight = new double[K];
		//Every proc get alpha
		for (int i = 1; i < numOfProcs; i++)
		{
			if (a <= aMax)
			{
				sendFileDataToSlave(N, K, LIMIT, input, signArray, i);
				sendTasksToSlave(K, a, weight, i);
				alphaIncrement(a, a0);
			}
		}

		//Receive
		int n = 1;
		while (n < numOfProcs)
		{
			//Recv the result of the task the slave did
			receiveResultsFromSlave(q, status, recvA, weight, bias, K, slaveNum);
			//Check if the slave found a classified q or the master has already a classified q
			if (q < QC || finalQ < QC || a > aMax)
			{
				terminateSlaves(recvA, finalA, a, slaveNum, finalWeight, weight, finalQ, q, finalBias, bias, K);
				n++;
			}
			//If the results not good enough - keep working
			else
			{
				initWeightsToZero(weight, K);
				sendTasksToSlave(K, a, weight, slaveNum);
				alphaIncrement(a, a0);
			}
		}
		char *filePathWrite = createFilePath(FOLDER, FILE_DEST_NAME);
		writeToFile(finalA, finalQ, finalWeight, finalBias, QC, startTime, K, filePathWrite);
		printResults(finalQ, finalA, finalWeight, finalBias, startTime, QC, K);
	}
	delete[] weight;
	delete[] finalWeight;
	delete[] input;
	delete[] signArray;
}
void sendFileDataToSlave(int N, int K, int LIMIT, double* input, int* signArray, int i)
{
	MPI_Send(&N, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(&K, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(&LIMIT, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(input, N*K, MPI_DOUBLE, i, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(signArray, N, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
}

//Return Quality of Classifier q according the formula - q = Nmis / N
double calcError(int N, int K, double *input, int* signArray, double *weight, double &bias, double& a, int LIMIT)
{
	int iter = 0;
	double sum = 0;
	for (; iter <= LIMIT;)
	{
		for (int i = 0; i < N; i++)
		{
			//With OMP
			sum = bias + calcSum(input, i, K, weight);
			//If points classified 
			if (checkSign(sum) != signArray[i])
			{
				//Only single thread works
				//fix weight ,count+1 and then restart
				fixWeights(weight, K, bias, a, &input[i*K], signArray[i]);
				iter++;
				break;
			}
		}
	}
	//count missclasified points and divieding by N
	if (iter > LIMIT)
	{
		return (double)countNmis(N, K, bias, input, signArray, weight);
	}
	else {
		return 0;
	}
}
double calcSum(double* input, int i, int K, double* weight)
{
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int j = 0; j < K; j++)
	{
		sum += input[i*(K)+j] * weight[j];
	}
	return sum;
}
int checkSign(double w)
{
	if (w >= 0)
		return 1;
	else return -1;
}

void fixWeights(double *weight, int size, double& bias, double a, double* cord, int error)
{
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		weight[i] += a * (error)* cord[i];
	}
	bias += error * a;
}

//Count how manny misses we got and return the number -OMP
int countNmis(int N, int K, double bias, double* input, int* signArray, double* weight)
{
	int count = 0;
	int* resultFromCuda = (int*)malloc(sizeof(int)* N);
	double sum;
	cudaError_t cudaStatus = calculateWithCuda(input, resultFromCuda, weight, N, K);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculateWithCuda failed!");
		return 1;
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	//With OMP
#pragma omp parallel for reduction(+:count)
	for (int i = 0; i < N; i++)
	{
		//With OMP
		sum = bias + calcSum(input, i, K, weight);
		//With Cuda
		if (resultFromCuda[i] != signArray[i])
		{
			count++;
		}
	}
	return count;
}

void sendTasksToSlave(int K, double& a, double * weight, int i)
{
	MPI_Send(&a, 1, MPI_DOUBLE, i, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(weight, K, MPI_DOUBLE, i, WORK_TAG, MPI_COMM_WORLD);
}

void sendResultsToMaster(double q, double a, double* weight, double bias, int K)
{
	MPI_Send(&q, 1, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(&a, 1, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(weight, K, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
	MPI_Send(&bias, 1, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD);
}

//Send terminate to Slaves
void terminateSlaves(double recvA, double& finalA, double& a, int slaveNum, double* finalWeight, double* weight, double& finalQ, double q, double& finalBias, double& bias, int K)
{
	MPI_Send(&a, 1, MPI_DOUBLE, slaveNum, TERMINATE_TAG, MPI_COMM_WORLD);
	//check if the slave found a smaller Alpha
	if (recvA < finalA)
	{
		finalA = recvA;
		for (int j = 0; j < K; j++)
			finalWeight[j] = weight[j];

		finalQ = q;
		finalBias = bias;
	}
}
void receiveResultsFromSlave(double& q, MPI_Status& status, double& recvA, double* weight, double& bias, int K, int& source)
{
	MPI_Recv(&q, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	source = status.MPI_SOURCE;
	MPI_Recv(&recvA, 1, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(weight, K, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&bias, 1, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}

void receiveDataToWorkFromMaster(MPI_Status& status, double& a, double& q, int& N, int& K, int& LIMIT, double*& input, int*& signArray, double*& weight)
{
	MPI_Recv(&N, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&K, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	weight = new double[K];
	input = new double[N*K];
	signArray = new int[N];
	MPI_Recv(&LIMIT, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(input, N*K, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(signArray, N, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&a, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}

//When get Work_Tag from master with new alpha to check
void slaveGotWork(MPI_Status& status, double& bias, int N, int K, double a, double q, int LIMIT, double* weight, double* input, int* signArray)
{
	bias = 0;
	MPI_Recv(weight, K, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	//Calc q
	q = (calcError(N, K, input, signArray, weight, bias, a, LIMIT) / N);
	//Send the results to master.
	sendResultsToMaster(q, a, weight, bias, K);
	//Wait for new task or terminate
	MPI_Recv(&a, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}
//Increment alpha
void alphaIncrement(double& a, double a0)
{
	a += a0;
}

//Initial all the weights to be 0
void initWeightsToZero(double* weightArray, int size)
{
	for (int i = 0; i < size; i++)
	{
		weightArray[i] = 0;
	}
}

void printWeights(double* finalWeight, double finalBias, int K)
{
	for (int i = 0; i < K; i++)
	{
		cout << finalWeight[i] << " ";
	}
	cout << finalBias << endl;

}

void printLinearEquation(double* weights, double bias, int K)
{
	cout << "java -jar BinaryClassification.jar " << FILE_SRC_NAME << " ";
	printWeights(weights, bias, K);
}
//Print the results- uses to print inside WMPI
void printResults(double finalQ, double finalA, double* finalWeight, double finalBias, double startTime, double QC, int K)
{
	if (finalQ < QC)
	{
		cout << "Alpha: " << finalA << " q: " << finalQ << endl;
		printLinearEquation(finalWeight, finalBias, K);
	}
	else
	{
		cout << "Alpha not found" << endl;
	}
	cout << "Runtime is: " << MPI_Wtime() - startTime << endl;
};
//Take folder and file name and make it root
char* createFilePath(char* folder, char* fileName)
{
	char *filePath = new char[strlen(folder) + strlen(fileName) + 1];
	strcpy(filePath, folder);
	strcat(filePath, fileName);
	return filePath;
}
bool overNumOfProcs(double a0, double aMax, int numOfProcs)
{
	if ((numOfProcs+1)*a0 > aMax)
	{
		return true; //means too much procs
	}
	return false;
}
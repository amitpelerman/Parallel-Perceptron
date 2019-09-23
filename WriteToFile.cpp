//Amit Pelerman
//203518006
#include "app.h"

void writeToFile(double finalA, double finalQ,  double* finalWeight, double finalBias, double QC, double startTime, int K, char* fileName)
{
	ofstream myFile;
	myFile.open(fileName);

	if (!myFile)
	{
		printf("ERROR via opening file\n");
		return;
	}
	if (finalQ < QC)
	{
		myFile << "Alpha: " << finalA << " q: " <<finalQ <<endl;
		for (int i = 0; i < K; i++)
		{
			myFile << finalWeight[i] << endl;
		}
		myFile << finalBias << endl;
	}
	else
	{
		myFile << "Alpha is not found"<< endl;
	}
	myFile.close();

}
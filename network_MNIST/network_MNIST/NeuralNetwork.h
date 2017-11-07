#pragma once
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

class NeuralNetwork {

public:
	NeuralNetwork(int numInput, int numHidden, int numOutput);
	void TrainNetwork(std::vector<std::vector<double>> trainData, int maxEprochs, double learnRate, double crossError);
	double Accuracy(std::vector<std::vector<double>> testData);
	std::vector <double>ComputeOutputs(std::vector <double> xValues);
private:
	int _numInput;
	int _numHidden;
	int _numOutput;

	std::vector<double> _inputs;
	std::vector<std::vector<double>> _hiddenWeights;
	std::vector<double> _hiddenBiases;
	std::vector<double> _hiddenOutputs;
	std::vector<std::vector<double>>  _outputWeights;
	std::vector<double> _outputBiases;
	std::vector<double> _outputs;
	std::vector <double> _hGrads; 
	std::vector <double> _oGrads;

	void InitializeWeights();
	static double HyperTanFunction(double x);
	static std::vector <double> Softmax(std::vector <double> oSums); 
	static void Shuffle(std::vector <int> sequence);
	double MeanCrossEntropyError(std::vector<std::vector<double>> trainData);
	int MaxIndex(std::vector<double> vec);
	void UpdateWeights(std::vector <double> tValues, double learnRate);
	void BackWard(std::vector <double> tValues, double learnRate);
	void ComputeGradient(std::vector <double> tValues);
	void UpdateBias(std::vector <double> tValues, double learnRate);
	std::vector <double> ComputeHiddenOutputs(std::vector <double> xValues);

};
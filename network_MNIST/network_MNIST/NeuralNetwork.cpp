#include "NeuralNetwork.h"
#define random (rand() / double(RAND_MAX)) / 100.0

NeuralNetwork::NeuralNetwork(int numInput, int numHidden, int numOutput) {
	_numInput = numInput;
	_numHidden = numHidden;
	_numOutput = numOutput;

	_hiddenWeights.resize(numInput);
	for (int i = 0; i < _hiddenWeights.size(); i++) {
		_hiddenWeights[i].resize(numHidden);
	}

	_outputWeights.resize(numHidden);
	for (int i = 0; i < _outputWeights.size(); i++) {
		_outputWeights[i].resize(numOutput);
	}

	_inputs.resize(numInput, 0);
	_hiddenBiases.resize(numHidden, 0);
	_hiddenOutputs.resize(numHidden, 0);
	_outputBiases.resize(numOutput, 0);
	_outputs.resize(numOutput, 0);
	_hGrads.resize(_numHidden, 0);
	_oGrads.resize(_numOutput, 0);

	InitializeWeights();
}

void NeuralNetwork::InitializeWeights() {
	int numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
	std::vector <double> initialWeight(numWeights, 0);
	srand(42);
	for (int i = 0; i < initialWeight.size(); ++i)
		initialWeight[i] = random;

	for (int i = 0; i < _numInput; i++) {
		for (int j = 0; j < _numHidden; j++) {
			_hiddenWeights[i][j] = random;
		}
	}

	for (int i = 0; i < _numHidden; i++) {
		_hiddenBiases[i] = random;
	}

	for (int i = 0; i < _numHidden; i ++) {
		for (int j = 0; j < _numOutput; j++) {
			_outputWeights[i][j] = random;
		}
	}

	for (int i = 0; i < _numOutput; i++) {
		_outputBiases[i] = random;
	}
}

std::vector <double> NeuralNetwork::ComputeHiddenOutputs(std::vector <double> xValues) {
	std::vector <double> hSums(_numHidden, 0);
	for (int i = 0; i < xValues.size(); i++) {
		_inputs[i] = xValues[i];
	}

	for (int j = 0; j < _numHidden; ++j) {
		for (int i = 0; i < _numInput; ++i) {
			hSums[j] += _inputs[i] * _hiddenWeights[i][j];
		}
	}

	for (int i = 0; i < _numHidden; i++) {
		hSums[i] += _hiddenBiases[i];
	}
	return hSums;
}

std::vector <double> NeuralNetwork::ComputeOutputs(std::vector <double> xValues) {

	std::vector <double> hSums;
	std::vector <double> oSums(_numOutput, 0);
	hSums = ComputeHiddenOutputs(xValues);

	for (int i = 0; i < _numHidden; i++) {
		_hiddenOutputs[i] = HyperTanFunction(hSums[i]);
	}

	for (int j = 0; j < _numOutput; j++) {
		for (int i = 0; i < _numHidden; i++) {
			oSums[j] += _hiddenOutputs[i] * _outputWeights[i][j];
		}
	}

	for (int i = 0; i < _numOutput; i++) {
		oSums[i] += _outputBiases[i];
	}

	std::vector <double> softOut = Softmax(oSums); 
	_outputs = softOut;

	return _outputs;
}

double NeuralNetwork::HyperTanFunction(double x) {
	if (x < -20.0) return -1.0;
	else if (x > 20.0) return 1.0;
	else return tanh(x);
}

std::vector <double> NeuralNetwork::Softmax(std::vector <double> oSums)  {
	double max = oSums[0];
	for (int i = 0; i < oSums.size(); i++) {
		if (oSums[i] > max) max = oSums[i];
	}

	double scale = 0.0;
	for (int i = 0; i < oSums.size(); i++) {
		scale += exp(oSums[i] - max);
	}

	std::vector <double> result(oSums.size(), 0);

	for (int i = 0; i < oSums.size(); i++) {
		result[i] = exp(oSums[i] - max) / scale;
	}
	return result; 
}

void NeuralNetwork::ComputeGradient(std::vector <double> tValues) {
	for (int i = 0; i < _oGrads.size(); i ++) {
		_oGrads[i] = (tValues[i] - _outputs[i]); // cross-entropy 
	}

	for (int i = 0; i < _hGrads.size(); i++) {
		double derivative = (1 - _hiddenOutputs[i]) * (1 + _hiddenOutputs[i]);
		double sum = 0.0;
		for (int j = 0; j < _numOutput; j++) {
			double x = _oGrads[j] * _outputWeights[i][j];
			sum += x;
		}
		_hGrads[i] = derivative * sum;
	}
}

void NeuralNetwork::UpdateWeights(std::vector <double> tValues, double learnRate) {
	for (int i = 0; i < _hiddenWeights.size(); ++i) {
		for (int j = 0; j < _hiddenWeights[0].size(); ++j) {
			double delta = learnRate * _hGrads[j] * _inputs[i];
			_hiddenWeights[i][j] += delta;
		}
	}
	for (int i = 0; i < _outputWeights.size(); ++i) {
		for (int j = 0; j < _outputWeights[0].size(); ++j) {
			double delta = learnRate * _oGrads[j] * _hiddenOutputs[i];
			_outputWeights[i][j] += delta;
		}
	}
}

void NeuralNetwork::UpdateBias(std::vector <double> tValues, double learnRate) {
	for (int i = 0; i < _hiddenBiases.size(); i ++) {
		double delta = learnRate * _hGrads[i] * 1.0;
		_hiddenBiases[i] += delta;
	}

	for (int i = 0; i < _outputBiases.size(); i++) {
		double delta = learnRate * _oGrads[i] * 1.0;
		_outputBiases[i] += delta;
	}
}

void NeuralNetwork::BackWard(std::vector <double> tValues, double learnRate) {
	ComputeGradient(tValues);
	UpdateWeights(tValues, learnRate);
	UpdateBias(tValues, learnRate);
}


double NeuralNetwork::MeanCrossEntropyError(std::vector <std::vector <double>> trainData)
{
	double sumError = 0.0;
	std::vector <double> xValues(_numInput, 0); 
	std::vector <double> tValues(_numOutput, 0); 

	for (int i = 0; i < trainData.size(); i++) 
	{
		for (int j = 0; j < xValues.size(); j++) {
			xValues[j] = trainData[i][j];
		}

		for (int j = 0; j < tValues.size(); j++) {
			tValues[j] = trainData[i][_numInput + j];
		}

		std::vector <double> yValues = ComputeOutputs(xValues); 
		for (int j = 0; j < _numOutput; j++) {
			sumError += log(yValues[j]) * tValues[j]; 
		}
	}
	return -1.0 * sumError / trainData.size();
}

void NeuralNetwork::TrainNetwork(std::vector<std::vector<double>> trainData, int maxEprochs, double learnRate, double crossError) {
	int epoch = 0;
	std::vector<double> xValues(_numInput, 0); 
	std::vector<double> tValues(_numOutput, 0); 

	std::vector<int> sequence(trainData.size(), 0);
	for (int i = 0; i < sequence.size(); i++)
		sequence[i] = i;

	for(epoch = 0; epoch < maxEprochs; epoch ++) {
		printf("NOW EPOCH = %i \n", epoch);
		double mcee = MeanCrossEntropyError(trainData);
		if (mcee < crossError) break;

		Shuffle(sequence); 
		for (int i = 0; i < trainData.size(); ++i) {
			int idx = sequence[i];
			for (int j = 0; j < xValues.size(); j++) {
				xValues[j] = trainData[idx][j];
			}

			for (int j = 0; j < tValues.size(); j++) {
				tValues[j] = trainData[idx][_numInput + j];
			}
			ComputeOutputs(xValues); 
			BackWard(tValues, learnRate);
		}
	}
} 

double NeuralNetwork::Accuracy(std::vector<std::vector<double>> testData) {
	int numCorrect = 0;
	int numWrong = 0;
	std::vector<double> xValues(_numInput, 0);
	std::vector<double> tValues(_numOutput, 0);
	std::vector<double> yValues;

	for (int i = 0; i < testData.size(); ++i) {

		for (int j = 0; j < xValues.size(); j++) {
			xValues[j] = testData[i][j];
		}

		for (int j = 0; j < tValues.size(); j++) {
			tValues[j] = testData[i][_numInput + j];
		}
		yValues = ComputeOutputs(xValues);
		int maxIndex = MaxIndex(yValues); 

		if (tValues[maxIndex] == 1.0) 
			numCorrect++;
		else
			numWrong++;
	}
	return (numCorrect * 1.0) / (numCorrect + numWrong); 
}

int NeuralNetwork::MaxIndex(std::vector<double> vec) {
	int bigIndex = 0;
	double biggestVal = vec[0];
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > biggestVal) {
			biggestVal = vec[i]; bigIndex = i;
		}
	}
	return bigIndex;
}

void NeuralNetwork::Shuffle(std::vector<int> sequence) {
	for (int i = 0; i < sequence.size(); i ++) {
		int r = i + (rand() % static_cast<int>((sequence.size() - 1) - i + 1));
		int tmp = sequence[r];
		sequence[r] = sequence[i];
		sequence[i] = tmp;
	}
}

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "ReadMinist.h"
#include "NeuralNetwork.h"

int main(int argc, char** argv) {

	if (argc < 5) {
		std::cout << "Error input_args: " << std::endl;
		std::cout << "1: Path to MNIST train-images " << std::endl;
		std::cout << "2: Path to MNIST train-labels " << std::endl;
		std::cout << "3: Path to MNIST test-images " << std::endl;
		std::cout << "4: Path to MNIST test-labels " << std::endl;
		std::cout << "5: number hidden neuron (default = 300) " << std::endl;
		std::cout << "6: maxEpochs (default = 25) " << std::endl;
		std::cout << "7: learnRate (default = 0.008) " << std::endl;
		std::cout << "8: crossError stop in train (default 0.005) " << std::endl;
		return 0;
	}

	std::string trainImageMNIST(argv[1]); 
	std::string trainLabelsMNIST(argv[2]); 
	std::string testImageMNIST(argv[3]); 
	std::string testLabelsMNIST(argv[4]); 

	int number_of_images = 60000;
	int image_size = 28 * 28;

	int maxEpochs = 25;
	double learnRate = 0.008;
	double crossError = 0.005;
	int numInput = 28 * 28;
	int numHidden = 300;
	int numOutput = 10;

	switch (argc) {
	case 6:
		numHidden = atoi(argv[5]);
		break;
	case 7:
		numHidden = atoi(argv[5]);
		maxEpochs = atoi(argv[6]);
		break;
	case 8:
		numHidden = atoi(argv[5]);
		maxEpochs = atoi(argv[6]);
		learnRate = atof(argv[7]);
		break;
	case 9:
		numHidden = atoi(argv[5]);
		maxEpochs = atoi(argv[6]);
		learnRate = atof(argv[7]);
		crossError = atof(argv[8]);
		break;
	default:
		break;
	}

	//std::string trainImageMNIST = "train-images-idx3-ubyte/train-images.idx3-ubyte";
	//std::string trainLabelsMNIST = "train-images-idx3-ubyte/train-labels.idx1-ubyte";
	//std::string testImageMNIST = "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
	//std::string testLabelsMNIST = "t10k-images-idx3-ubyte/t10k-labels.idx1-ubyte";

	std::vector<std::vector<double>> trainData_MNIST;
	read_Mnist(trainImageMNIST, trainData_MNIST);

	std::vector<double> vec_labels(number_of_images);
	read_Mnist_Label(trainLabelsMNIST, vec_labels);

	std::vector<std::vector<double>> trainLabel_MNIST;
	trainLabel_MNIST.resize(number_of_images);
	for (int i = 0; i < trainLabel_MNIST.size(); i++) {
		trainLabel_MNIST[i] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		trainLabel_MNIST[i][(int)vec_labels[i]] = 1.0;
	}

	trainData_MNIST.resize(number_of_images);
	for (int i = 0; i < number_of_images; i++) {
		for (int k = 0; k < 10; k++) {
			trainData_MNIST[i].push_back(trainLabel_MNIST[i][k]);
		}
	}

	std::vector<std::vector<double>> testData_MNIST;
	read_Mnist(testImageMNIST, testData_MNIST);

	int number_of_test_images = 10000;
	std::vector<double> vec_test_labels(number_of_test_images);
	read_Mnist_Label(testLabelsMNIST, vec_test_labels);
	std::vector<std::vector<double>> testLabel_MNIST;
	testLabel_MNIST.resize(number_of_test_images);
	for (int i = 0; i < testLabel_MNIST.size(); i++) {
		testLabel_MNIST[i] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		testLabel_MNIST[i][(int)vec_test_labels[i]] = 1.0;
	}

	testData_MNIST.resize(number_of_test_images);
	for (int i = 0; i < number_of_test_images; i++) {
		for (int k = 0; k < 10; k++) {
			testData_MNIST[i].push_back(testLabel_MNIST[i][k]);
		}
	}

	NeuralNetwork nn = NeuralNetwork(numInput, numHidden, numOutput);

	nn.TrainNetwork(trainData_MNIST, maxEpochs, learnRate, crossError);

	double trainAcc = nn.Accuracy(trainData_MNIST);
	double testAcc = nn.Accuracy(testData_MNIST);

	std::cout << "!!!  result train  " << trainAcc << " result test " << testAcc << std::endl;

	return 0;
}
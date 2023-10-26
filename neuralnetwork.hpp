#pragma once

#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

class NeuralNetwork
{
public:
	enum ActivationFunction
	{
		HYPERBOLIC_TANGENT_ACTIVATION,
		SIGMOID_ACTIVATION,
		RELU_ACTIVATION,
		SOFTMAX_ACTIVATION
	};

	NeuralNetwork();
	NeuralNetwork(const std::vector<unsigned int>& topology, NeuralNetwork::ActivationFunction hiddenActivation = RELU_ACTIVATION, NeuralNetwork::ActivationFunction outputActivation = SOFTMAX_ACTIVATION);
	NeuralNetwork(const NeuralNetwork& neuralNetwork);

	const std::vector<unsigned int>& getTopology() const;
	double getEvaluation() const;
	void setEvaluation(double evaluation);
	NeuralNetwork::ActivationFunction getHiddenActivation();
	NeuralNetwork::ActivationFunction getOutputActivation();

	void mutate(double mutationRate = 0.2, double mutationScale = 0.5);
	std::vector<double> calculateOutput(std::vector<double> inputs);

protected:
	class Neuron
	{
	public:
		Neuron();
		Neuron(unsigned int connectionsIn, bool initializeValues = true);
		Neuron(const Neuron& neuron);

		double getActivation(const std::vector<double>& inputs, NeuralNetwork::ActivationFunction activationFunc);

		std::vector<double> weightsIn;
		double bias;
	};

	class Layer
	{
	public:
		Layer(unsigned int numberOfNeurons);
		Layer(unsigned int numberOfNeurons, std::shared_ptr<Layer> previous, bool initializeValues = true);
		Layer(const Layer& layer);

		std::vector<double> getActivations(const std::vector<double>& inputs, NeuralNetwork::ActivationFunction activationFunc);

		std::vector<std::shared_ptr<Neuron>> neurons;
		std::shared_ptr<Layer> previousLayer;
	};

private:
	static std::mt19937& getGenerator();
	static time_t& getSeed();
	static void setSeed(const time_t& s);

	void serialize(std::ofstream& outputStream);
	void deserialize(const std::string& input);

	std::vector<std::shared_ptr<Layer>> layers;
	std::vector<unsigned int> topology;
	double evaluation;
	NeuralNetwork::ActivationFunction hiddenActivation;
	NeuralNetwork::ActivationFunction outputActivation;

	friend class Generation;
};

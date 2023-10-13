#include "neuralnetwork.hpp"

#include <sstream>

static time_t seed = std::time(NULL);
static std::mt19937 generator(static_cast<unsigned int>(seed));

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int>& topology, NeuralNetwork::ActivationFunction hiddenActivation, NeuralNetwork::ActivationFunction outputActivation) : topology(topology), evaluation(0.0), hiddenActivation(hiddenActivation), outputActivation(outputActivation)
{
	if (topology.size() < 2)
		throw std::runtime_error("Network must have an input and output layer");

	this->layers.push_back(std::make_shared<Layer>(topology[0]));
	
	for (int i = 1; i < topology.size(); i++)
	{
		if (topology[i] == 0)
			throw std::runtime_error("Layer must have at least 1 neuron");

		this->layers.push_back(std::make_shared<Layer>(topology[i], this->layers.back()));
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
{
	this->topology = neuralNetwork.topology;
	this->hiddenActivation = neuralNetwork.hiddenActivation;
	this->outputActivation = neuralNetwork.outputActivation;
	this->evaluation = neuralNetwork.evaluation;

	this->layers.push_back(std::make_shared<Layer>(*neuralNetwork.layers[0]));

	for (int i = 1; i < neuralNetwork.layers.size(); i++)
	{
		std::shared_ptr<Layer> lp = std::make_shared<Layer>(*neuralNetwork.layers[i]);
		lp->previousLayer = this->layers.back();

		this->layers.push_back(lp);
	}
}

const std::vector<unsigned int>& NeuralNetwork::getTopology() const
{
	return this->topology;
}

double NeuralNetwork::getEvaluation() const
{
	return this->evaluation;
}

void NeuralNetwork::setEvaluation(double evaluation)
{
	this->evaluation = evaluation;
}

NeuralNetwork::ActivationFunction NeuralNetwork::getHiddenActivation()
{
	return this->hiddenActivation;
}

NeuralNetwork::ActivationFunction NeuralNetwork::getOutputActivation()
{
	return this->outputActivation;
}

void NeuralNetwork::mutate(double mutationRate, double mutationScale)
{
	if (mutationRate < 0.0 || mutationScale < 0.0)
		throw std::runtime_error("Mutation rate and mutation scale must be non-negative");

	std::normal_distribution<double> distr1(0.0 - mutationScale, mutationScale);
	std::uniform_real_distribution<double> distr2(0.0, 1.0);

	for (int l = 0; l < this->layers.size(); l++)
		for (int n = 0; n < this->layers[l]->neurons.size(); n++)
		{
			for (int w = 0; w < this->layers[l]->neurons[n]->weightsIn.size(); w++)
				if (distr2(generator) <= mutationRate)
					this->layers[l]->neurons[n]->weightsIn[w] += distr1(generator);

			if (distr2(generator) <= mutationRate)
				this->layers[l]->neurons[n]->bias += distr1(generator);
		}
}

std::vector<double> NeuralNetwork::getOutput(std::vector<double> inputs)
{
	for (int i = 1; i < this->layers.size() - 1; i++)
		inputs = this->layers[i]->getActivations(inputs, this->hiddenActivation);

	return this->layers.back()->getActivations(inputs, this->outputActivation);
}

std::mt19937& NeuralNetwork::getGenerator()
{
	return generator;
}

time_t& NeuralNetwork::getSeed()
{
	return seed;
}

void NeuralNetwork::setSeed(const time_t& s)
{
	seed = s;
	generator = std::mt19937(static_cast<unsigned int>(seed));
}

void NeuralNetwork::serialize(std::ofstream& outputStream)
{
	if (!outputStream.good())
		throw std::runtime_error("An error occured while opening the file");

	outputStream << this->evaluation << " "
		<< this->hiddenActivation << " "
		<< this->outputActivation << " "
		<< this->getTopology().size();

	for (unsigned int t : this->getTopology())
		outputStream << " " << t;

	for (int l = 1; l < this->layers.size(); l++)
		for (const std::shared_ptr<NeuralNetwork::Neuron>& neuron : this->layers[l]->neurons)
		{
			for (const double& w : neuron->weightsIn)
				outputStream << " " << w;

			outputStream << " " << neuron->bias;
		}

	outputStream << std::endl;
}

void NeuralNetwork::deserialize(const std::string& input)
{
	this->topology.clear();
	this->layers.clear();

	std::stringstream strStream(input);
	int iToken;
	unsigned int uiToken;
	double dToken;

	strStream >> this->evaluation;
	strStream >> iToken;
	this->hiddenActivation = static_cast<ActivationFunction>(iToken);
	strStream >> iToken;
	this->outputActivation = static_cast<ActivationFunction>(iToken);
	strStream >> iToken;

	for (int i = 0; i < iToken; i++)
	{
		strStream >> uiToken;
		this->topology.push_back(uiToken);
	}

	if (topology.size() < 2)
		throw std::runtime_error("Network must have an input and output layer");

	this->layers.push_back(std::make_shared<Layer>(topology[0]));

	for (int i = 1; i < topology.size(); i++)
	{
		if (topology[i] == 0)
			throw std::runtime_error("Layer must have at least 1 neuron");

		this->layers.push_back(std::make_shared<Layer>(topology[i], this->layers.back(), false));

		for (unsigned int n = 0; n < topology[i]; n++)
		{
			for (unsigned int w = 0; w < topology[static_cast<size_t>(i - 1)]; w++)
			{
				strStream >> dToken;
				this->layers.back()->neurons[n]->weightsIn[w] = dToken;
			}

			strStream >> dToken;
			this->layers.back()->neurons[n]->bias = dToken;
		}
	}

	if (strStream.fail())
		throw std::runtime_error("Failed to read in values");
}

NeuralNetwork::Layer::Layer(unsigned int numberOfNeurons) : previousLayer(nullptr)
{
	for (unsigned int i = 0; i < numberOfNeurons; i++)
		this->neurons.push_back(std::make_shared<Neuron>());
}

NeuralNetwork::Layer::Layer(unsigned int numberOfNeurons, std::shared_ptr<Layer> previous, bool initalizeWeights)
{
	for (unsigned int i = 0; i < numberOfNeurons; i++)
		this->neurons.push_back(std::make_shared<Neuron>(previous->neurons.size(), initalizeWeights));

	this->previousLayer = previous;
}

NeuralNetwork::Layer::Layer(const Layer& layer) : previousLayer(nullptr)
{
	for (const std::shared_ptr<Neuron>& neuron : layer.neurons)
		this->neurons.push_back(std::make_shared<Neuron>(*neuron));
}

std::vector<double> NeuralNetwork::Layer::getActivations(const std::vector<double>& inputs, NeuralNetwork::ActivationFunction activationFunc)
{
	std::vector<double> activations;

	for (int i = 0; i < this->neurons.size(); i++)
		activations.push_back(this->neurons[i]->getActivation(inputs, activationFunc));

	if (activationFunc == SOFTMAX_ACTIVATION) // Softmax Activations
	{
		double max = *std::max_element(activations.begin(), activations.end());
		double sum = 0.0;

		for (int i = 0; i < activations.size(); i++)
		{
			activations[i] = std::exp(activations[i] - max);
			sum += activations[i];
		}

		for (int i = 0; i < activations.size(); i++)
			activations[i] /= sum;
	}

	return activations;
}

NeuralNetwork::Neuron::Neuron() : bias(0.0) {}

NeuralNetwork::Neuron::Neuron(unsigned int connectionsIn, bool initializeWeights)
{
	// Xavier Weight Initialisation
	std::uniform_real_distribution<double> distribution(0.0 - (1.0 / std::sqrt(connectionsIn)), 1.0 / std::sqrt(connectionsIn));

	for (unsigned int i = 0; i < connectionsIn; i++)
		if (initializeWeights)
			this->weightsIn.push_back(distribution(generator));
		else
			this->weightsIn.push_back(0.0);

	if (initializeWeights)
		this->bias = distribution(generator);
	else
		this->bias = 0.0;
}

NeuralNetwork::Neuron::Neuron(const Neuron& neuron)
{
	this->bias = neuron.bias;
	this->weightsIn = neuron.weightsIn;
}

double NeuralNetwork::Neuron::getActivation(const std::vector<double>& inputs, NeuralNetwork::ActivationFunction activationFunc)
{
	double activation = this->bias;

	for (int i = 0; i < this->weightsIn.size(); i++)
	{
		if (this->weightsIn.size() != inputs.size())
			throw std::runtime_error("Invalid number of inputs");

		activation += this->weightsIn[i] * inputs[i];
	}

	if (activationFunc == RELU_ACTIVATION) // Rectified Linear Unit (ReLU) Activation
		return std::fmax(0.0, activation);
	if (activationFunc == SIGMOID_ACTIVATION) // Sigmoid Activation
		return 1.0 / (1.0 + std::exp(-activation));

	return activation;
}

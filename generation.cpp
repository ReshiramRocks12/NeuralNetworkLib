#include "generation.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>

Generation::Generation(unsigned int nNetworks, std::vector<unsigned int> topology, unsigned int nBatches, NeuralNetwork::ActivationFunction hiddenActivation, NeuralNetwork::ActivationFunction outputActivation) : generationNum(0), nBatches(nBatches)
{
	for (unsigned int i = 0; i < nNetworks; i++)
		this->networks.push_back(std::make_shared<NeuralNetwork>(topology, hiddenActivation, outputActivation));
}

Generation::Generation() : generationNum(0), nBatches(0) {}

std::shared_ptr<Generation> Generation::createNewGeneration(double mutationRate, double mutationScale, Generation::SelectionAlgoritm selectionAlgo, double selectionPercent, Generation::CrossoverAlgorithm crossoverAlgo, double crossOverRate, unsigned int kPoints)
{
	if (this->networks.size() == 0)
		throw std::runtime_error("Generation is empty");

	std::vector<unsigned int> sortedIndices;
	std::vector<unsigned int> selectedIndices;

	this->sortByEvaluation(sortedIndices);

	switch (selectionAlgo)
	{
	case FITNESS_PROPORTIONATE_SELECTION:
		selectedIndices = fitnessProportionateSelection(selectionPercent);
		break;
	case TOURNAMENT_SELECTION:
		// TODO: Implement
		throw "Not Implemented";
		break;
	case ELITISM:
		for (unsigned int i = 0; i < sortedIndices.size() * selectionPercent; i++)
			selectedIndices.push_back(sortedIndices[i]);
		break;
	}

	NeuralNetwork::setSeed(NeuralNetwork::getSeed() + 1);
	std::shared_ptr<Generation> generation = std::make_shared<Generation>();
	generation->generationNum = this->generationNum + 1;
	generation->nBatches = this->nBatches;

	if (selectedIndices.size() % 2)
		for (int i = 0; i < sortedIndices.size(); i++)
			if (std::find(selectedIndices.begin(), selectedIndices.end(), sortedIndices[i]) == selectedIndices.end())
			{
				selectedIndices.push_back(sortedIndices[i]);
				break;
			}

	for (int i = 0; i < std::ceil(1.0 / selectionPercent); i++)
		for (unsigned int index : selectedIndices)
		{
			generation->networks.push_back(std::make_shared<NeuralNetwork>(*this->networks[index]));

			if (generation->networks.size() == this->networks.size())
				break;
		}

	std::shuffle(generation->networks.begin(), generation->networks.end(), NeuralNetwork::getGenerator());

	switch (crossoverAlgo)
	{
	case SINGLE_POINT_CROSSOVER:
		// TODO: Implement
		throw "Not Implemented";
		break;
	case K_POINT_CROSSOVER:
		// TODO: Implement
		throw "Not Implemented";
		break;
	case UNIFORM_CROSSOVER:
		generation->uniformCrossover(crossOverRate);
		break;
	}

	for (std::shared_ptr<NeuralNetwork>& n : generation->networks)
		n->mutate(mutationRate, mutationScale);

	return generation;
}

std::vector<std::shared_ptr<NeuralNetwork>> Generation::getBatch(unsigned int batchNum)
{
	if (this->networks.size() == 0)
		throw std::runtime_error("Generation is empty");
	if (batchNum > this->nBatches)
		throw std::runtime_error("Batch size out of range");

	if (this->nBatches == 0)
		return this->networks;

	unsigned int batchSize = static_cast<unsigned int>(this->networks.size() / nBatches);

	if ((static_cast<unsigned long long>(batchNum + 2)) * batchSize > this->networks.size())
		return std::vector<std::shared_ptr<NeuralNetwork>>(this->networks.begin() + (batchNum * batchSize), this->networks.end());

	return std::vector<std::shared_ptr<NeuralNetwork>>(this->networks.begin() + (batchNum * batchSize), this->networks.begin() + ((batchNum + 1) * batchSize));
}

void Generation::serialize(const std::string& folder, bool sort)
{
	if (this->networks.size() == 0)
		throw std::runtime_error("Generation is empty");

	std::ofstream outputStream(folder + "\\gen" + std::to_string(this->generationNum) + ".txt");

	if (!outputStream.good())
		throw std::runtime_error("An error occured while opening the file");

	std::vector<unsigned int> sortedIndices;

	if (sort)
		this->sortByEvaluation(sortedIndices);
	
	outputStream << std::setprecision(17);
	outputStream << this->generationNum << " " << this->networks.size() << " " << this->nBatches << " " << NeuralNetwork::getSeed() << std::endl;

	if (sortedIndices.size() > 0)
		for (int i = 0; i < sortedIndices.size(); i++)
		{
			outputStream << sortedIndices[i] << " ";
			this->networks[sortedIndices[i]]->serialize(outputStream);
		}
	else
		for (int i = 0; i < this->networks.size(); i++)
		{
			outputStream << i << " ";
			this->networks[i]->serialize(outputStream);
		}

	outputStream.close();
}

void Generation::deserialize(const std::string& file)
{
	std::ifstream inputStream(file);

	if (!inputStream.good())
		throw std::runtime_error("An error occured while opening the file");

	std::string line;
	unsigned int ln = 2;
	unsigned int genNum, genSize, nBatches, index;
	time_t seed;

	std::getline(inputStream, line);
	std::stringstream strStream(line);

	strStream >> genNum;
	strStream >> genSize;
	strStream >> nBatches;
	strStream >> seed;

	this->generationNum = genNum;
	this->nBatches = nBatches;
	NeuralNetwork::setSeed(seed);

	if (strStream.fail())
		throw std::runtime_error("Invalid syntax at line 1");

	this->networks = std::vector<std::shared_ptr<NeuralNetwork>>(genSize);
	
	for (unsigned int i = 0; i < genSize; i++)
	{
		try
		{
			inputStream >> index;
			std::getline(inputStream, line);
			this->networks[index] = std::make_shared<NeuralNetwork>();
			this->networks[index]->deserialize(line);
		}
		catch (std::runtime_error& e)
		{
			throw std::runtime_error("Invalid syntax at line " + std::to_string(ln) + ": " + e.what());
		}

		ln++;
	}
}

void Generation::sortByEvaluation(std::vector<unsigned int>& sorted)
{
	for (unsigned int i = 0; i < this->networks.size(); i++)
		sorted.push_back(i);

	std::sort(sorted.begin(), sorted.end(), [this](unsigned int a, unsigned int b)
	{
		return this->networks[a]->getEvaluation() > this->networks[b]->getEvaluation();
	});
}

std::vector<unsigned int> Generation::fitnessProportionateSelection(double selectionPercent)
{
	unsigned int selectionNum = static_cast<unsigned int>(this->networks.size() * selectionPercent);
	std::vector<unsigned int> selectedIndexes;

	double totalEval = 0.0;

	for (std::shared_ptr<NeuralNetwork>& network : this->networks)
		totalEval += network->getEvaluation();

	double selectionEval;
	std::uniform_real_distribution<double> distribution(0.0, totalEval);

	while (selectedIndexes.size() < selectionNum)
	{
		selectionEval = distribution(NeuralNetwork::getGenerator());
		
		for (int i = 0; i < this->networks.size(); i++)
		{
			selectionEval -= this->networks[i]->getEvaluation();

			if (selectionEval <= 0.0)
				if (std::find(selectedIndexes.begin(), selectedIndexes.end(), i) == selectedIndexes.end())
				{
					selectedIndexes.push_back(i);
					break;
				}
		}
	}

	return selectedIndexes;
}

void Generation::uniformCrossover(double crossOverRate)
{
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	double temp;

	for (int i = 1; i < this->networks.size(); i += 2)
		for (int l = 0; l < this->networks[i]->layers.size(); l++)
			for (int n = 0; n < this->networks[i]->layers[l]->neurons.size(); n++)
			{
				for (int w = 0; w < this->networks[i]->layers[l]->neurons[n]->weightsIn.size(); w++)
				{
					if (distribution(NeuralNetwork::getGenerator()) < crossOverRate)
					{
						temp = this->networks[i]->layers[l]->neurons[n]->weightsIn[w];
						this->networks[i]->layers[l]->neurons[n]->weightsIn[w] = this->networks[static_cast<size_t>(i - 1)]->layers[l]->neurons[n]->weightsIn[w];
						this->networks[static_cast<size_t>(i - 1)]->layers[l]->neurons[n]->weightsIn[w] = temp;
					}
				}

				if (distribution(NeuralNetwork::getGenerator()) < crossOverRate)
				{
					temp = this->networks[i]->layers[l]->neurons[n]->bias;
					this->networks[i]->layers[l]->neurons[n]->bias = this->networks[static_cast<size_t>(i - 1)]->layers[l]->neurons[n]->bias;
					this->networks[static_cast<size_t>(i - 1)]->layers[l]->neurons[n]->bias = temp;
				}
			}
}

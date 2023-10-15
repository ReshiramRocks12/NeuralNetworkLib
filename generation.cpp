#include "generation.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <random>

Generation::Generation(unsigned int nNetworks, std::vector<unsigned int> topology, unsigned int nBatches, NeuralNetwork::ActivationFunction hiddenActivation, NeuralNetwork::ActivationFunction outputActivation) : generationNum(0), nBatches(nBatches)
{
	this->networks.reserve(nNetworks);

	for (unsigned int i = 0; i < nNetworks; i++)
		this->networks.push_back(std::make_shared<NeuralNetwork>(topology, hiddenActivation, outputActivation));
}

Generation::Generation() : generationNum(0), nBatches(0) {}

std::shared_ptr<Generation> Generation::createNewGeneration(double mutationChance, double mutationScale, Generation::SelectionAlgoritm selectionAlgo, double selectionPercent, double kPercent, double p, Generation::CrossoverAlgorithm crossoverAlgo, double crossOverRate, unsigned int kPoints)
{
	if (this->networks.size() == 0)
		throw std::runtime_error("Generation is empty");
	if (mutationChance < 0.0 || mutationChance > 1.0)
		throw std::runtime_error("Mutation chance must be in the range of 0 to 1");
	if (selectionPercent < 0.0 || selectionPercent > 1.0)
		throw std::runtime_error("Selection percent must be in the range of 0 to 1");
	if (kPercent < 0.0 || kPercent > 1.0)
		throw std::runtime_error("K percent must be in the range of 0 to 1");

	std::vector<unsigned int> sortedIndices(this->networks.size());
	std::vector<unsigned int> selectedIndices;

	std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
	this->sortByEvaluation(sortedIndices);
	selectedIndices.reserve(static_cast<size_t>(this->networks.size() * selectionPercent));

	switch (selectionAlgo)
	{
	case FITNESS_PROPORTIONATE_SELECTION:
		fitnessProportionateSelection(selectionPercent, selectedIndices);
		break;
	case TOURNAMENT_SELECTION:
		tournamentSelection(selectionPercent, kPercent, 1.0, selectedIndices);
		break;
	case PROBABILISTIC_TOURNAMENT_SELECTION:
		tournamentSelection(selectionPercent, kPercent, p, selectedIndices);
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
		generation->kPointCrossover(1);
		break;
	case K_POINT_CROSSOVER:
		generation->kPointCrossover(kPoints);
		break;
	case UNIFORM_CROSSOVER:
		generation->uniformCrossover(crossOverRate);
		break;
	}

	for (std::shared_ptr<NeuralNetwork>& n : generation->networks)
		n->mutate(mutationChance, mutationScale);

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

	if ((static_cast<size_t>(batchNum + 2)) * batchSize > this->networks.size())
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
	outputStream << this->generationNum << " " << this->networks.size() << " " << this->nBatches << " " << (NeuralNetwork::getSeed() - this->generationNum) << std::endl;

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
	NeuralNetwork::setSeed(seed + this->generationNum);

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
	std::sort(sorted.begin(), sorted.end(), [this](unsigned int a, unsigned int b)
	{
		return this->networks[a]->getEvaluation() > this->networks[b]->getEvaluation();
	});
}

void Generation::fitnessProportionateSelection(double selectionPercent, std::vector<unsigned int>& selectedIndexes)
{
	unsigned int selectionNum = static_cast<unsigned int>(this->networks.size() * selectionPercent);
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
}

void Generation::tournamentSelection(double selectionPercent, double kPercent, double p, std::vector<unsigned int>& selectedIndexes)
{
	unsigned int selectionNum = static_cast<unsigned int>(this->networks.size() * selectionPercent);
	unsigned int k = static_cast<unsigned int>(this->networks.size() * kPercent);
	std::vector<unsigned int> tournamentIndexes;
	tournamentIndexes.reserve(k);

	std::uniform_int_distribution<unsigned int> distribution1(0, this->networks.size() - 1);
	std::uniform_real_distribution<double> distribution2(0.0, 1.0);
	int si;

	while (selectedIndexes.size() < selectionNum)
	{
		tournamentIndexes.clear();

		while (tournamentIndexes.size() < k)
		{
			si = distribution1(NeuralNetwork::getGenerator());

			if (std::find(selectedIndexes.begin(), selectedIndexes.end(), si) == selectedIndexes.end() && std::find(tournamentIndexes.begin(), tournamentIndexes.end(), si) == tournamentIndexes.end())
				tournamentIndexes.push_back(si);
		}

		sortByEvaluation(tournamentIndexes);

		for (int i = 0; i < tournamentIndexes.size(); i++)
			if (distribution2(NeuralNetwork::getGenerator()) < p * std::pow(1 - p, i))
				selectedIndexes.push_back(tournamentIndexes[i]);
	}
}

void Generation::uniformCrossover(double crossOverRate)
{
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	
	for (size_t i = 1; i < this->networks.size(); i += 2)
		for (int l = 0; l < this->networks[i]->layers.size(); l++)
			for (int n = 0; n < this->networks[i]->layers[l]->neurons.size(); n++)
			{
				for (int w = 0; w < this->networks[i]->layers[l]->neurons[n]->weightsIn.size(); w++)
					if (distribution(NeuralNetwork::getGenerator()) < crossOverRate)
						std::swap(this->networks[i]->layers[l]->neurons[n]->weightsIn[w], this->networks[i - 1]->layers[l]->neurons[n]->weightsIn[w]);

				if (distribution(NeuralNetwork::getGenerator()) < crossOverRate)
					std::swap(this->networks[i]->layers[l]->neurons[n]->bias, this->networks[i - 1]->layers[l]->neurons[n]->bias);
			}
}

void Generation::kPointCrossover(unsigned int k)
{
	unsigned int total = 1;
	size_t iC = 0;
	bool crossover = false;

	for (size_t i = 1; i < this->networks[0]->topology.size(); i++)
		total += this->networks[0]->topology[i - 1] * this->networks[0]->topology[i] + this->networks[0]->topology[i];

	std::vector<unsigned int> kIndexes(total);
	std::iota(kIndexes.begin(), kIndexes.end(), 0);
	std::shuffle(kIndexes.begin(), kIndexes.end(), NeuralNetwork::getGenerator());
	kIndexes.resize(k);
	std::sort(kIndexes.begin(), kIndexes.end());

	for (size_t i = 1; i < this->networks.size(); i += 2)
		for (int l = 0; l < this->networks[i]->layers.size(); l++)
			for (int n = 0; n < this->networks[i]->layers[l]->neurons.size(); n++)
			{
				for (int w = 0; w < this->networks[i]->layers[l]->neurons[n]->weightsIn.size(); w++)
				{
					if (std::find(kIndexes.begin(), kIndexes.end(), iC) != kIndexes.end())
						crossover = !crossover;

					if (crossover)
						std::swap(this->networks[i]->layers[l]->neurons[n]->weightsIn[w], this->networks[i - 1]->layers[l]->neurons[n]->weightsIn[w]);

					iC++;
				}

				if (std::find(kIndexes.begin(), kIndexes.end(), iC) != kIndexes.end())
					crossover = !crossover;

				if (crossover)
					std::swap(this->networks[i]->layers[l]->neurons[n]->bias, this->networks[i - 1]->layers[l]->neurons[n]->bias);

				iC++;
			}
}

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuralnetwork.hpp"

class Generation
{
public:
	enum SelectionAlgoritm
	{
		FITNESS_PROPORTIONATE_SELECTION,
		TOURNAMENT_SELECTION,
		PROBABILISTIC_TOURNAMENT_SELECTION,
		ELITISM
	};

	enum CrossoverAlgorithm
	{
		SINGLE_POINT_CROSSOVER,
		K_POINT_CROSSOVER,
		UNIFORM_CROSSOVER
	};

	Generation(unsigned int nNetworks, std::vector<unsigned int> topology, unsigned int nBatches = 0, NeuralNetwork::ActivationFunction hiddenActivation = NeuralNetwork::ActivationFunction::RELU_ACTIVATION, NeuralNetwork::ActivationFunction outputActivation = NeuralNetwork::ActivationFunction::SOFTMAX_ACTIVATION);
	Generation();
	std::shared_ptr<Generation> createNewGeneration(
		double mutationRate = 0.2, double mutationScale = 0.5,
		Generation::SelectionAlgoritm selectionAlgo = FITNESS_PROPORTIONATE_SELECTION, double selectionPercent = 0.5, double kPercent = 0.2, double p = 0.9,
		Generation::CrossoverAlgorithm crossoverAlgo = UNIFORM_CROSSOVER, double crossOverRate = 0.5, unsigned int kPoints = 0
	);
	std::vector<std::shared_ptr<NeuralNetwork>> getBatch(unsigned int batchNum);
	std::vector<std::shared_ptr<NeuralNetwork>> getAllNetworks();
	unsigned int getBatchNum();
	unsigned int getBatchSize();
	unsigned int getGenerationNum();
	
	void serialize(const std::string& folder, bool sort = true);
	void deserialize(const std::string& file);

private:
	void sortByEvaluation(std::vector<unsigned int>& sortedIndices);

	void fitnessProportionateSelection(double selectionPercent, std::vector<unsigned int>& selectedIndexes);
	void tournamentSelection(double selectionPercent, double kPercent, double p, std::vector<unsigned int>& selectedIndexes);

	void uniformCrossover(double crossOverRate);
	void kPointCrossover(unsigned int k);

	std::vector<std::shared_ptr<NeuralNetwork>> networks;
	unsigned int generationNum;
	unsigned int nBatches;
};

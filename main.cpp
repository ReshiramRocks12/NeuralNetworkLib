#include <iostream>
#include <vector>

#include "generation.hpp"
#include "neuralnetwork.hpp"

int main()
{
	Generation gen(100, { 10, 8, 4 });
	gen.serialize("./Test");
	gen.createNewGeneration()->serialize("./Test");

	Generation genCopy;

	try
	{
		genCopy.deserialize("./Test/gen0.txt");
	}
	catch (std::runtime_error& e)
	{
		std::cout << e.what() << std::endl;
	}

	genCopy.serialize("./Test2");
	genCopy.createNewGeneration()->serialize("./Test2");

	return 0;
}

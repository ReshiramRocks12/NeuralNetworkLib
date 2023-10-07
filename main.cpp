#include <iostream>
#include <vector>

#include "generation.hpp"
#include "neuralnetwork.hpp"

int main()
{
	Generation gen(10, { 11, 8, 4 });
	gen.serialize("./Test");
	
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

	return 0;
}

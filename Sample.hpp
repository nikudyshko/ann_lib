#ifndef SAMPLE_H 
#define SAMPLE_H 

//necessary includes, armadillo - for linear algebra 
#include <armadillo> 

//implements Sample structure, that represents train or validation sample 
struct Sample
{ 
	//in - input data 
	//out - corresponding to in output data 
	arma::vec in, out; 
	//~Sample() - destructor, that clears memory of vectors 
	~Sample()
	{
		in.clear(); 
		out.clear(); 
	}
};

#endif 

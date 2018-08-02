#ifndef SAMPLE_H 
#define SAMPLE_H 

//necessary includes, armadillo - for linear algebra 
#include <armadillo> 

//implements Sample class that represents train or validation sample 
class Sample
{ 
public: 
	//in - input data 
	//out - corrseponding to in output data 
	arma::vec in, out; 
	//~Sample() - destructor that clears memory of vectors 
	~Sample()
	{
		in.clear(); 
		out.clear(); 
	}
}; 

#endif 

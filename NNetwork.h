#ifndef NNETWORK_H
#define NNETWORK_H 

#ifdef __BORLANDC__ 
#include <float.h> 
#define isfinite _finite 
#define isinf _isinf 
#define isnan _isnan 
#endif 

class NNetwork
{
private: 
public: 
	NNetwork(std::vector< std::pair< size_t, std::string > > &architect_); 
	~NNetwork(); 
};

#endif

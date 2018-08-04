#ifndef ACTIVATION_H 
#define ACTIVATION_H 

#include <cmath> 

class Functor 
{
public: 
	virtual double operator()(double x) { return 0.0; }; 
}; 

class ASigmoid : public Functor
{
public: 
	virtual double operator()(double x)
	{
		if (x > 300.0) return 1.0; 
		else if (x < -300.0) return 0.0; 
		else return 1.0 / (1.0 + std::exp(-x)); 
	}
}; 

class ATanh : public Functor
{
public: 
	virtual double operator()(double x)
	{
		if (x > 300.0) return 1.0; 
		else if (x < -300.0) return -1.0; 
		else return std::tanh(x); 
	}
}; 

class AReLU : public Functor
{
public: 
	virtual double operator()(double x)
	{
		return (x >= 0.0) ? x : 0.0; 
	} 
}; 

class AExp : public Functor
{
public: 
	double max_val; 
	virtual double operator()(double x)
	{
		return std::exp(x - max_val); 
	} 
	AExp() : max_val(0.0) {}; 
};

#endif 

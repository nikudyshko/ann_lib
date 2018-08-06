#ifndef ACTIVATION_H 
#define ACTIVATION_H 

#include <iostream> 
#include <string> 
#include <exception> 
#include <cmath> 

#include <armadillo> 

class Functor 
{
public: 
	double max_val; 
	virtual double operator()(double x) { return 0.0; }; 
	Functor() : max_val(0.0) {}; 
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
	virtual double operator()(double x)
	{
		return std::exp(x - this->max_val); 
	} 
	AExp() : Functor() {}; 
}; 

class Activation
{ 
public: 
	virtual void operator()(arma::vec const &logit, arma::vec &activ) {}; 
}; 

class _ASigm : public Activation 
{ 
private: 
	Functor f; 
public: 
	virtual void operator()(arma::vec const &logit, arma::vec &activ)
	{
		activ = logit; 
		activ.transform(this->f); 
	} 
	_ASigm() : f(ASigmoid()) {}; 
}; 

class _ATanh : public Activation
{ 
private: 
	Functor f; 
public: 
	virtual void operator()(arma::vec const &logit, arma::vec &activ)
	{ 
		activ = logit; 
		activ.transform(this->f); 
	} 
	_ATanh() : f(ATanh()) {}; 
}; 

class _AReLU : public Activation
{
private: 
	Functor f; 
public: 
	virtual void operator()(arma::vec const &logit, arma::vec &activ)
	{
		activ = logit; 
		activ.transform(this->f); 
	} 
	_AReLU() : f(AReLU()) {}; 
}; 

class _ASoftMax : public Activation
{
private: 
	Functor f; 
public: 
	virtual void operator()(arma::vec const &logit, arma::vec &activ)
	{ 
		activ = logit; 
		this->f.max_val = arma::max(activ); 
		activ.transform(this->f); 

		double sum = arma::accu(activ); 
		activ /= sum; 
	} 
	_ASoftMax() :f (AExp()) {}; 
}; 

class Derivative
{
public: 
	virtual void operator()(arma::vec const &activ, arma::mat &jacoby) {}; 
}; 

class _DSigm : public Derivative
{
public: 
	virtual void operator()(arma::vec const &activ, arma::mat &jacoby)
	{
		for(size_t i = 0; i < jacoby.n_rows; ++i) 
			jacoby.at(i, i) = activ.at(i)*(1.0 - activ.at(i)); 
	}
}; 

class _DTanh : public Derivative
{ 
public: 
	virtual void operator()(arma::vec const &activ, arma::mat &jacoby)
	{
		for(size_t i = 0; i < jacoby.n_rows; ++i) 
			jacoby.at(i, i) = 1.0 - activ.at(i)*activ.at(i); 
	} 
}; 

class _DReLU : public Derivative
{ 
public: 
	virtual void operator()(arma::vec const &activ, arma::mat &jacoby)
	{
		for(size_t i = 0; i < jacoby.n_rows; ++i) 
			jacoby.at(i, i) = (activ.at(i) > 0) ? 1.0 : 0.0; 
	} 
}; 

class _DSoftMax : public Derivative
{
public: 
	virtual void operator()(arma::vec const &activ, arma::mat &jacoby)
	{
		for(size_t i = 0; i < jacoby.n_rows; ++i) 
		{
			jacoby.at(i, i) = activ.at(i)*(1.0 - activ.at(i)); 
			for (size_t j = i + 1; j < jacoby.n_cols; ++j)
				jacoby.at(i, j) = -activ.at(i)*activ.at(j); 
		} 
	}
}; 

class Functions
{
public: 
	static Activation activative(std::string const &fun_name) 
	{
		if (!fun_name.compare("sigm")) return _ASigm(); 
		else if (!fun_name.compare("tanh")) return _ATanh(); 
		else if (!fun_name.compare("relu")) return _AReLU(); 
		else if (!fun_name.compare("softmax")) return _ASoftMax(); 

		std::cerr << "Invalid function name!" << std::endl; 
		throw std::runtime_error("Invalid function name!"); 
	} 
	static Derivative derivative(std::string const &fun_name) 
	{
		if (!fun_name.compare("sigm")) return _DSigm(); 
		else if (!fun_name.compare("tanh")) return _DTanh(); 
		else if (!fun_name.compare("relu")) return _DReLU(); 
		else if (!fun_name.compare("softmax")) return _DSoftMax(); 

		std::cerr << "Invalid function name!" << std::endl; 
		throw std::runtime_error("Invalid function name!"); 
	} 
};

#endif 

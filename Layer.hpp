#ifndef LAYER_H 
#define LAYER_H 

#include <ctime> 
#include <random>
#include <armadillo> 
#include "Activation.hpp"

class Loss
{
public: 
	double loss; 

	arma::vec error; 

	virtual void forward(arma::vec const &net_output, arma::vec const &true_output); 
	virtual void backward(arma::vec const &net_output, arma::vec const &true_output); 
};

class CrossEntropyLoss : public Loss 
{ 
public: 
	double loss; 

	arma::vec error; 

	virtual void forward(arma::vec const &net_output, arma::vec const &true_output); 
	virtual void backward(arma::vec const &net_output, arma::vec const &true_output); 

	CrossEntropyLoss(size_t num_neurons); 
}; 

CrossEntropyLoss::CrossEntropyLoss(size_t num_neurons) 
{ 
	this->error.zeros(num_neurons); 
}

void CrossEntropyLoss::forward(arma::vec const &net_output, arma::vec const &true_output) 
{
	this->loss = 0.0; 
	for (size_t i = 0; i < net_output.n_elem; ++i)
		this->loss += true_output.at(i)*std::log(net_output.at(i)) + (1.0 - true_output.at(i))*std::log(1.0 - net_output.at(i)); 
} 

void CrossEntropyLoss::backward(arma::vec const &net_output, arma::vec const &true_output) 
{ 
	for (size_t i = 0; i < error.n_elem; ++i)
		this->error.at(i) = (true_output.at(i) - net_output.at(i)) / (net_output.at(i)*(1.0 - net_output.at(i))); 
} 

class MSELoss : public Layer
{ 
private: 
public: 
	double loss; 

	arma::vec error; 

	virtual void forward(arma::vec const &net_output, arma::vec const &true_output); 
	virtual void backward(arma::vec const &net_output, arma::vec const &true_output); 

	MSELoss(size_t num_neurons); 
}; 

MSELoss::MSELoss(size_t num_neurons) 
{ 
	this->error.zeros(num_neurons); 
}

void MSELoss::forward(arma::vec const &net_output, arma::vec const &true_output)
{ 
	this->loss = 0.0; 
	for(size_t i = 0; i < net_output.n_elem; ++i) 
		this->loss += std::pow((net_output.at(i) - true_output.at(i)), 2.0); 
	loss *= 2.0; 
} 

void MSELoss::backward(arma::vec const &net_output, arma::vec const &true_output) 
{ 
	for (size_t i = 0; i < this->error.n_elem; ++i)
		this->error.at(i) = (net_output.at(i) - true_output.at(i)); 
}

#endif 

#pragma once 

#ifndef LAYER_H 
#define LAYER_H 

#include <ctime> 
#include <random>
#include <armadillo> 
#include "Activation.hpp"

class Dense
{ 
private: 
	friend class NNetwork; 
	size_t num_neurons; 
	const double bias_const = 0.1; 
	arma::vec biases, grad_b; 
	arma::mat weights, grad_w; 
	arma::mat jacoby; 

	Activation activative; 
	Derivative derivative; 

	void zero_weights(); 
	void init_weights(); 
public: 
	arma::vec input, error; 
	arma::vec logit, activ, delta; 
	void forward(arma::vec const &input_); 
	void backward(arma::vec const &error_); 

	Dense(size_t previous_num_neurons, size_t current_num_neurons, std::string const &fun_name); 
	~Dense(); 
}; 

Dense::Dense(size_t previous_num_neurons, size_t current_num_neurons, std::string const &fun_name)
{
	this->num_neurons = current_num_neurons;
	this->input.zeros(previous_num_neurons);
	this->error.zeros(current_num_neurons);
	this->logit.zeros(current_num_neurons);
	this->activ.zeros(current_num_neurons);
	this->delta.zeros(current_num_neurons);
	this->biases.zeros(current_num_neurons);
	this->grad_b.zeros(current_num_neurons);
	this->weights.zeros(current_num_neurons, previous_num_neurons);
	this->grad_w.zeros(current_num_neurons, previous_num_neurons);
	this->jacoby.zeros(current_num_neurons, current_num_neurons); 
	this->activative = Functions::activative(fun_name); 
	this->derivative = Functions::derivative(fun_name); 
} 

Dense::~Dense() 
{
	this->input.clear(); 
	this->error.clear(); 
	this->logit.clear(); 
	this->activ.clear(); 
	this->delta.clear(); 
	this->biases.clear(); 
	this->grad_b.clear(); 
	this->weights.clear(); 
	this->grad_w.clear(); 
	this->jacoby.clear(); 
}

void Dense::init_weights()
{ 
	std::mt19937 gen(time(NULL)); 
	std::normal_distribution<double> dist(0.0, std::sqrt(1 / static_cast<double>(this->weights.n_cols))); 
	this->biases.fill(this->bias_const); 
	this->weights.imbue(dist(gen)); 
} 

void Dense::zero_weights()
{ 
	this->grad_b.zeros(); 
	this->grad_w.zeros(); 
} 

void Dense::forward(arma::vec const &input_)
{ 
	this->input = input_; 
	this->logit = this->weights*this->input + this->biases; 
	this->activative(this->logit, this->activ); 
} 

void Dense::backward(arma::vec const &error_)
{
	this->derivative(this->activ, this->jacoby);
	this->delta = this->jacoby*error_;
	this->error = this->weights.t()*this->delta; 
	this->grad_b += this->delta; 
	this->grad_w += this->delta*this->input.t(); 
} 

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

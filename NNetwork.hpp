#pragma once 

#ifndef NNETWORK_H
#define NNETWORK_H 

//necessary includes and defines for Embarcadero C++ Builder compatibility 
#ifdef __BORLANDC__ 
#include <float.h> 
#define isfinite _finite 
#define isinf _isinf 
#define isnan _isnan 
#endif 

//necessary includes, armadillo - for linear algebra stuff 
#include <vector> 
#include <string> 
#include <armadillo> 
//Sample - for sample structure 
#include "Layer.hpp" 
#include "Sample.hpp" 

class HyperP
{
public: 
	size_t num_neurons; 
	std::string fun_name; 
	~HyperP() { this->fun_name.clear(); }
};

//implementing of neural network 
class NNetwork
{
private: 
	static constexpr double b1 = 0.9, b2 = 0.999, eps = 1e-8; 
	static constexpr double bias_const = 0.1; 
	//num_layers - number of layers in neural network 
	//train_set_size - size of the train set :) 
	//validation_set_size - size of the validation set :) 
	size_t num_layers, train_set_size, validation_set_size; 
	//architect - contains description of neural network's architecture 
	//in form (number of neurons, name of activation function) 
	std::default_random_engine shuffle_gen, rand_gen; 
	std::uniform_int_distribution<size_t> dist_tr, dist_vl; 

	std::vector< HyperP > architect; 
	//train_set - array of samples for training neural network 
	//validation_set - array of samples for testing neural network 
	std::vector< Sample* > train_set, validation_set; 
	std::vector< Activation > a_funs; 
	std::vector< Derivative > d_funs; 

	Loss loss_fun; 

	arma::field< arma::vec > logits, activs, deltas; 
	arma::field< arma::vec > biases, grad_b; 
	arma::field< arma::vec > b_m, b_m_old; 
	arma::field< arma::vec > b_v, b_v_old; 
	arma::field< arma::mat > weights, grad_w; 
	arma::field< arma::mat > w_m, w_m_old; 
	arma::field< arma::mat > w_v, w_v_old; 
	arma::field< arma::mat > jacobies; 

	Loss loss_fun; 

	//zero_gradient() - fills gradient fields with zeros 
	void zero_gradient(); 
	//feed_forward() - calculates forward propagation of full network 
	void feed_forward(arma::vec const &in); 
	//back_propagation() - calculates backward propagation of full network 
	void back_propagation(arma::vec const &out); 
	//update_weights() - updates weights and biases according to ADAM algorithm 
	void update_weights(int iter, size_t batch_size, double learning_rate, double reg_rate); 

	//function that loads dataset. Used by load_train_set and load_validation_set 
	std::vector< Sample > load_data_set(std::string const &file_name); 

	//function that clears memory dedicated to neural network
	void clear_neural_net(); 
	//function that clears memory dedicated to train_set 
	void clear_train_set(); 
	//function that clears memory dedicated to validation_set 
	void clear_validation_set(); 
public: 
	//init_weights() - initializes weights and biases of network 
	void init_weights(); 

	//train_step() - makes a train step 
	void train_step(int iter, size_t batch_num, size_t batch_size, double learning_rate, double reg_rate); 
	//predict() - gives prediction of neural network using provided input 
	arma::vec& predict(arma::vec const &in); 

	//calc_accuracy() - calculates accuracy of neural network's prediction (on full_set or on single batch) 
	//uses validation_set 
	double calc_accuracy(size_t batch_size, bool full_set = false); 
	//calc_loss() - calculates loos of neural network (on full_set or on single batch); 
	double calc_loss(size_t batch_size, bool full_set = false); 

	//save_neural_net() - saves neural network to txt-file 
	void save_neural_net(std::string const &file_name); 
	//load_neural_net() - loads neural network from txt-file 
	void load_neural_net(std::string const &file_name); 

	//load_train_set() - loads train set :) 
	void load_train_set(std::string const &file_name); 
	//load_validation_set() - loads validation set :) 
	void load_validation_set(std::string const &file_name); 

	//NNetwork(vector) - class constructor, uses array, that describes architecture of neural network 
	NNetwork(std::vector< std::pair< size_t, std::string > > &architect_); 
	//NNetwork(strinf) - class constructor, uses file name to load architecture and weights of neural network from file 
	NNetwork(std::string const &file_name); 
	//~NNetwork() - class destructor, frees memory dedicated to neural network 
	~NNetwork(); 
}; 

void NNetwork::zero_gradient()
{ 
	for (size_t i = 1; i < this->num_layers; ++i)
	{ 
		this->grad_b[i].zeros(); 
		this->grad_w[i].zeros(); 
	} 
} 

void NNetwork::feed_forward(arma::vec const &in)
{ 
	this->activs[0] = in; 
	for (size_t i = 1; i < this->num_layers; ++i)
	{
		this->logits[i] = this->weights[i] * this->activs[i - 1] + this->biases[i]; 
		this->a_funs[i](this->logits[i], this->activs[i]); 
	} 
} 

void NNetwork::back_propagation(arma::vec const &out)
{ 
	this->loss_fun.backward(this->activs[this->num_layers - 1], out); 
	this->d_funs[this->num_layers - 1](this->activs[this->num_layers - 1], this->jacobies[this->num_layers - 1]); 
	this->deltas[this->num_layers - 1] = this->jacobies[this->num_layers - 1] * this->loss_fun.error; 
	this->grad_b[this->num_layers - 1] = this->deltas[this->num_layers - 1]; 
	this->grad_w[this->num_layers - 1] = this->deltas[this->num_layers - 1] * this->activs[this->num_layers - 2]; 
	for (size_t i = this->num_layers - 2; i >= 1; --i)
	{ 
		this->d_funs[i](this->activs[i], this->jacobies[i]); 
		this->deltas[i] = this->jacobies[i] * this->weights[i + 1].t()*this->deltas[i + 1]; 
		this->grad_b[i] += this->deltas[i]; 
		this->grad_w[i] += this->deltas[i] * this->activs[i - 1].t(); 
	} 
} 

void NNetwork::clear_neural_net()
{ 
	this->logits.clear(); 
	this->activs.clear(); 
	this->deltas.clear(); 
	this->biases.clear(); 
	this->grad_b.clear(); 
	this->b_m.clear(); 
	this->b_v.clear(); 
	this->b_m_old.clear(); 
	this->b_v_old.clear(); 
	this->weights.clear(); 
	this->grad_w.clear(); 
	this->w_m.clear(); 
	this->w_v.clear(); 
	this->w_m_old.clear(); 
	this->w_v_old.clear(); 
	this->jacobies.clear(); 

	this->architect.erase(this->architect.begin(), this->architect.end()); 
} 

void NNetwork::clear_train_set()
{
	this->train_set.erase(this->train_set.begin, this->train_set.end()); 
} 

void NNetwork::clear_validation_set()
{
	this->validation_set.erase(this->validation_set.begin(), this->validation_set.end()); 
} 

void NNetwork::init_weights()
{ 
	std::mt19937 gen(time(NULL)); 

	for (size_t i = 1; i < this->num_layers; ++i)
	{
		std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / this->architect[i].num_neurons)); 
		this->weights[i].imbue([&gen, &dist]() {return dist(gen); }); 
		this->biases[i].fill(this->bias_const); 
	} 
} 

void NNetwork::train_step(int iter, size_t batch_num, size_t batch_size, double learning_rate, double reg_rate) 
{ 
	this->zero_gradient(); 
	for (size_t i = 0; i < batch_size; ++i)
	{
		this->feed_forward(this->train_set[batch_num*batch_size + i]->in); 
		this->back_propagation(this->validation_set[batch_num*batch_size + i]->out); 
	} 
	this->update_weights(iter, batch_size, learning_rate, reg_rate); 
} 

arma::vec& NNetwork::predict(arma::vec const &in)
{ 
	this->feed_forward(in); 
	return this->activs[this->num_layers - 1]; 
} 

double NNetwork::calc_accuracy(size_t batch_size, bool full_set = false) 
{ 
	size_t begin = 0, end = 0; 
	double acc = 0.0; 

	if (full_set)
	{
		begin = 0; 
		end = this->validation_set_size; 
	} 
	else
	{
		begin = this->dist_vl(rand_gen);
		end = begin + batch_size;
	} 

	for (size_t i = begin; i < end; ++i)
	{
		this->feed_forward(this->validation_set[i]->in); 
		if (this->activs[this->num_layers - 1].index_max() == this->validation_set[i]->out.index_max())
			acc += 1.0; 
	} 

	return acc / static_cast<double>(end - begin); 
} 

double NNetwork::calc_loss(size_t batch_size, bool full_set = false) 
{ 
	size_t begin = 0, end = 0; 
	double loss = 0.0; 

	if (full_set)
	{
		begin = 0; 
		end = this->train_set_size; 
	} 
	else
	{
		begin = this->dist_tr(rand_gen); 
		end = begin + batch_size; 
	} 

	for (size_t i = begin; i < end; ++i)
	{
		this->feed_forward(this->train_set[i]->in); 
		this->loss_fun.forward(this->activs[this->num_layers - 1], this->train_set[i]->out); 
		loss += this->loss_fun.loss; 
	} 

	return loss / static_cast<double>(end - begin); 
}

#endif

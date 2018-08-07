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

//implementing of neural network 
class NNetwork
{
private: 
	//num_layers - number of layers in neural network 
	//train_set_size - size of the train set :) 
	//validation_set_size - size of the validation set :) 
	size_t num_layers, train_set_size, validation_set_size; 
	//architect - contains description of neural network's architecture 
	//in form (number of neurons, name of activation function) 
	std::vector< std::pair< size_t, std::string > > architect; 
	//network - contains a Layers of neural network; 
	std::vector< Dense > network; 
	Loss loss_fun; 
	//train_set - array of samples for training neural network 
	//validation_set - array of samples for testing neural network 
	std::vector< Sample > train_set, validation_set; 

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
	for (size_t i = 0; i < this->num_layers; ++i)
		this->network[i].zero_weights(); 
} 

void NNetwork::feed_forward(arma::vec const &in)
{ 
	this->network[0].forward(in); 
	for (size_t i = 1; i < this->num_layers; ++i)
		this->network[i].forward(this->network[i - 1].activ); 
} 

void NNetwork::back_propagation(arma::vec const &out)
{
	this->loss_fun.backward(this->network[this->num_layers - 1].activ, out); 
	this->network[this->num_layers - 1].backward(this->loss_fun.backward); 
	for (size_t i = this->num_layers - 2; i >= 0; --i)
		this->network[i].backward(this->network[i + 1].error); 
} 

void NNetwork::clear_neural_net()
{ 
	this->network.erase(this->network.begin(), this->network.end()); 
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
	for (size_t i = 0; i < this->num_layers; ++i)
		this->network[i].init_weights(); 
} 

void NNetwork::train_step(int iter, size_t batch_num, size_t batch_size, double learning_rate, double reg_rate) 
{
	this->zero_gradient(); 
	for(size_t i = 0; i < batch_size; i++) 
	{
		this->feed_forward(this->train_set[batch_num*batch_size + i].in); 
		this->back_propagation(this->train_set[batch_num*batch_size + i].out); 
	} 
	this->update_weights(iter, batch_size, learning_rate, reg_rate); 
} 

arma::vec& NNetwork::predict(arma::vec const &in)
{
	this->feed_forward(in); 
	return this->network[this->num_layers - 1].activ; 
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
		std::default_random_engine gen(time(NULL)); 
		std::uniform_int_distribution<size_t> dist(0, this->validation_set_size - batch_size - 1); 

		begin = dist(gen); 
		end = begin + batch_size; 
	} 

	for (size_t i = 0; i < this->validation_set_size; ++i)
	{
		this->feed_forward(this->validation_set[i].in); 
		if (this->network[this->num_layers - 1].activ.index_max() == this->validation_set[i].out.index_max())
			acc += 1.0; 
	} 

	return acc / static_cast<double>(this->validation_set_size); 
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
		std::default_random_engine gen(time(NULL)); 
		std::uniform_int_distribution<size_t> dist(0, this->train_set_size - batch_size - 1); 

		begin = dist(gen); 
		end = begin + batch_size; 
	} 

	for (size_t i = 0; i < this->train_set_size; ++i)
	{
		this->feed_forward(this->train_set[i].in); 
		this->loss_fun.forward(this->network[this->num_layers - 1].activ, this->train_set[i].out); 
		loss += this->loss_fun.loss; 
	} 

	return loss / static_cast<double>(this->train_set_size); 
}

#endif

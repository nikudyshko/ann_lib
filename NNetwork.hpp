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
	std::vector< Layer > network; 
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

	//train_step() - makes a train step (on full_epoch or single batch) 
	void train_step(int iter, size_t batch_size, double learning_rate, double reg_rate, bool full_epoch = false); 
	//predict() - gives prediction of neural network using provided input 
	arma::vec& predict(arma::vec const &in); 

	//calc_accuracy() - calculates accuracy of neural network's prediction (on full_set or on single batch) 
	//uses validation_set 
	double calc_accuracy(bool full_set = false); 
	//calc_loss() - calculates loos of neural network (on full_set or on single batch); 
	double calc_loss(bool full_set = false); 

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

#endif

#ifndef LAYER_H 
#define LAYER_H 

#include <armadillo> 

class Layer
{ 
public: 
	virtual void forward(arma::vec const &input_) {}; 
	virtual void backward(arma::vec const &error_) {}; 
}; 

class Dense : public Layer
{ 
private: 
	size_t num_neurons; 
	arma::vec biases, grad_b; 
	arma::mat weights, grad_w; 

	void init_weights(); 
public: 
	arma::vec input, error; 
	arma::vec logit, activ, delta; 

	virtual void forward(arma::vec const &input_); 
	virtual void backward(arma::vec const &error_); 
};

#endif 

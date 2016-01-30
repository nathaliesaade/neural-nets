import numpy as np
class NN:
    """"Neural Networks class"""
    def __init__(self, cost):
        """takes in cost as argument (MSE (mean-squared error) or cross entropy)"""
        self.cost = cost

        # number of nodes in layers
        self.n_i = 784  # +1 for bias
        self.n_h = 200  # +1 for bias 
        self.n_o = 10
        
        self.i=0

        ##learning rate
        self.epsilon = 0.001 
        
        self.exp_threshold = 100
        

        # initialize node-activations
        self.xi_0, self.xi_1, self.xi_2 = [],[],[]
        self.xi_0 = np.ones((self.n_i,1)).ravel().reshape((self.n_i,1))
        self.xi_1 = np.ones((self.n_h,1)).ravel().reshape((self.n_h,1))
        self.xi_2= np.ones((self.n_o,1)).ravel().reshape((self.n_o,1))

        #initialize node weight matrices
        self.w_1 = np.random.uniform(-0.01,0.01,(self.n_i+1, self.n_h)) #784+1 by 200
        self.w_2 = np.random.uniform(-0.01,0.01,(self.n_h+1, self.n_o)) #200+1 by  10
    
    def forwardPropagate(self, inputs):
        """Performs forward propagation taking as argument the inputs
        and returning the outputs"""
        self.xi_0 = inputs
        self.xi_0  = np.array(self.xi_0).reshape((self.n_i,1))
        self.xi_0 = np.vstack((1,self.xi_0))
    
        self.Si_1 = np.multiply(self.w_1, self.xi_0)
        self.Si_1 = (np.sum(self.Si_1, axis=0))
        
        self.xi_1 = np.tanh(self.Si_1)
        self.xi_1 = np.hstack((1, self.xi_1))
        self.xi_1 = np.reshape(self.xi_1, (self.xi_1.shape[0],1))

        self.Si_2 = np.multiply(self.w_2, self.xi_1)
        self.Si_2 = (np.sum(self.Si_2, axis=0))
        
        self.xi_2 = 1/(1+np.exp(-self.Si_2))
            
        return self.xi_2
      
    def backPropagate (self, targets):
        """Performs back propagation taking as an argument the targets
        and returns the new weights"""
        
        ##compute deltas for output layer
        if(self.cost=="MSE"): ##Mean-Squared Error as loss function
            d_error = np.array(self.xi_2.ravel()-np.array(targets)).reshape((self.n_o,1))
        else:
            #"CROSS ENTROPY as loss function"
            d_error = np.array([-1*(np.array(targets)[i]/(self.xi_2[i]))+(1-np.array(targets)[i])/(1-self.xi_2[i]) for i in range(self.n_o)]).reshape((self.n_o,1))

        
        pr_2 = self.xi_2.ravel().T *(1- self.xi_2.ravel().T)
        
        d_error = np.array(d_error.ravel()).reshape((1,self.n_o)).flatten()
        delta_i_2 = np.multiply(d_error,pr_2).reshape((self.n_o,1))

        ##deltas for hidden layer     
        err = np.dot(self.w_2,delta_i_2)
        pr_1 = 1-self.xi_1**2
        delta_i_1 = np.multiply(err,pr_1)
        
        #####Update the weights
        ######update output weights w_2
        self.w_2 = self.w_2 - self.epsilon*np.dot(self.xi_1,delta_i_2.T)
                
        ######update input weights w_1
        self.w_1 = self.w_1 - self.epsilon*np.dot(self.xi_0,delta_i_1.T[:,1:])

        return self.w_1, self.w_2


    def train(self, images, labels, max_iterations):
        """trains the neural network"""
        cost_values_train = []
        cost_values_valid = []
        acc_values_train = []
        acc_values_valid = []

    
        while (self.i<max_iterations): 
            self.i+=1
            #pick random data point (x,y) from all data
            ix = np.random.choice(range(images.shape[0]))
            inputs = images[ix]
            targets = labels[ix]

            self.forwardPropagate(inputs)
            self.backPropagate(targets) 
                
        return self.w_1, self.w_2
    
    def predict(self, testImages):
        """making predictions using the weights found after training"""
        images = testImages
        #compute labels of all images using the weights
        predictions = []
        for p in images:
            predictions.append(self.forwardPropagate(p))
        return predictions
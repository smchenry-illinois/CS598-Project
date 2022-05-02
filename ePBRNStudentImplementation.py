#####################################################
#   CS598 DL4H Spring 2022 Reproducibility Project  #
# ------------------------------------------------- #
#   ePBRNStudentImplementation.py                   #
# ------------------------------------------------- #
#   Group ID: 72, ID: Paper 252                     #
#   Steve McHenry, William Plefka                   #
#   {mchenry7, wplefka2}@illinois.edu               #
#===================================================#
#   Module containing the student implementation    #
#   for the ePBRN dataset (Scheme B)                #
#####################################################

# Import the torch library which we'll use for our implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold

# CS598 Project Code / Support Vector Machine / Implementation
class ePBRNReproducerSVM(nn.Module):
    def __init__(self, num_features, inverse_reg=0.0):
        # Create the our PyTorch support vector machine model
        super(ePBRNReproducerSVM, self).__init__()

        # STEP 1
        # Specify parameters for our PyTorch SVM model based upon the analogous
        # parameters used by the original paper's sklearn SVC

        # PyTorch SVM concept                           Analogous sklearn SVC parameter
        # -------------------                           -------------------------------
        self.inverse_reg = inverse_reg                  # C (inverse of the regularization strength)
        #                                               # kernel (original paper uses linear, as do we)

        # STEP 2
        # Define the layers for our lr model
        self.num_input_features = num_features

        self.fc1 = nn.Linear(in_features=self.num_input_features, out_features=1, bias=False)

        # STEP 3
        # Define the criteria and optimizer
        self.num_max_epochs = 1000
        self.criterion = nn.HingeEmbeddingLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
            lr=0.001,
            weight_decay=inverse_reg)

    def forward(self, x):
        # Perform a forward pass on the nn; it is not recommended to call this
        # function directly, but to instead call fit(...) or predict(...) so that model's
        # mode is correctly set automatically
        x = self.fc1(x)

        return torch.squeeze(x)

    def fit(self, X_train, y_train):
        # Train the SVM with the specified parameters; analogous to sklearn's
        # SVC.fit(...) method
        self.train()

        loss_previous_epoch = 1.0
        loss_consecutive_epochs_minimal = 0

        for epoch_i in np.arange(self.num_max_epochs):
            loss = None
            kfold = KFold(n_splits=10, shuffle=True, random_state=12345)

            for train_indicies, _ in kfold.split(X_train):
                self.optimizer.zero_grad()
                output = self.forward(X_train[train_indicies])
                output *= -1
                loss = self.criterion(output, y_train[train_indicies])
                loss.backward()
                self.optimizer.step()

            # Determine if criteria for early training termination is satisfied
            if (np.abs(loss_previous_epoch - loss.item())) <= 0.0001:
                loss_consecutive_epochs_minimal = loss_consecutive_epochs_minimal + 1

                if(loss_consecutive_epochs_minimal == 50):
                    break
            else:
                loss_consecutive_epochs_minimal = 0

            loss_previous_epoch = loss.item()

    def predict(self, X_test):
        # Test the nn with the specified parameters; analogous to sklearn's
        # SVC.predict(...) method
        self.eval()
        return self.forward(X_test)

# CS598 Project Code / Neural Network Model / Implementation
class ePBRNReproducerNN(nn.Module):
    def __init__(self, num_features, weight_decay=0.0):
        # Create the our PyTorch nn model
        super(ePBRNReproducerNN, self).__init__()

        # STEP 1
        # Specify parameters for our PyTorch nn model based upon the analogous
        # parameters used by the original paper's sklearn MLPClassifier

        # PyTorch nn concept                            Analogous sklearn MLPClassifier parameter
        # ------------------                            -----------------------------------------
        self.optimizer = None                           # solver (optimizer; original paper uses LBFGS, but we will use SGD (defined later) due to PyTorch-sklearn differences)
        self.optimizer_weight_decay = weight_decay      # alpha (L2 penalty/regularization term)
        self.num_hidden_layer_nodes = 64                # hidden_layer_sizes (tuple of hidden layer nodes; original paper uses 256, see "[2]" below in Step 2)
        self.activation = F.relu                        # activation (activation function)
        self.random_state = 12345                       # random_state (static, random state for reproducibility)
        #                                               # batch_size (minibatch size; unused in our model)
        #                                               # learning_rate (tells the model to use the provided initial learning rate; n/a to our model)
        self.optimizer_learning_rate_init = 0.05        # learning_rate_init (initial learning rate; original paper uses 0.001; see "[1]" below)
        self.optimizer_dampening = 0.0                  # power_t (dampening)
        self.num_max_epochs = 3000                      # max_iter (maximum number of epochs when using stochastic optimizers)
        self.shuffle = True                             # shuffle (shuffle samples in each iteration)
        self.tolerance = 0.0001                         # tol (optimization tolorance; early training termination)
        #                                               # verbose (print model progress debug messages to console; specified by unused by original paper)
        #                                               # warm_start (initialize the model with the results of previous executions; specified but unused by original paper)
        self.optimizer_momentum = 0.9                   # momentum (optimizer momentum)
        self.use_nesterov_momentum = True               # nesterovs_momentum (use Nesterov's momentum in the optimizer)
        #                                               # early_stopping (terminate early when validation is not improving; 'False' in original paper)
        #                                               # validation_fraction (validation data set criteria for early stopping; specified by unused by original paper)
        #                                               # beta_1 (parameter for Adam optimizer; specified but unused by original paper)
        #                                               # beta_2 (parameter for Adam optimizer; specified but unused by original paper)
        #                                               # epsilon (parameter for Adam optimizer; specified but unused by original paper)

        # [1] The original authors' model used a learning rate of 0.001, however, we discovered that with our
        #     model's adjusted hyperparameters, a slightly larger learning rate resulted in a measurably higher
        #     sensitivity score of approximately 3%

        # STEP 2
        # Define the layers for our nn model
        # [2] The original authors' model uses a single hidden layer of 256 nodes, but we found this decision
        #     to be suboptimal; we determined a much more optimal architecture using two hidden layers, with
        #     64 and 16 nodes, respectively
        self.num_input_features = num_features

        self.fc1 = nn.Linear(in_features=self.num_input_features, out_features=self.num_hidden_layer_nodes, bias=False)
        self.fc2 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.fc3 = nn.Linear(in_features=16, out_features=1, bias=False)

        # STEP 3
        # Define the criteria and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
            lr=self.optimizer_learning_rate_init,
            weight_decay=self.optimizer_weight_decay,
            momentum=self.optimizer_momentum,
            dampening=self.optimizer_dampening,
            nesterov=self.use_nesterov_momentum)

    def forward(self, x):
        # Perform a forward pass on the nn; it is not recommended to call this
        # function directly, but to instead call fit(...) or predict(...) so that model's
        # mode is correctly set automatically
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return torch.squeeze(x)

    def fit(self, X_train, y_train):
        # Train the nn with the specified parameters; analogous to sklearn's
        # MLPClassifier.fit(...) method
        self.train()
    
        loss_previous_epoch = 1.0
        loss_consecutive_epochs_minimal = 0

        for epoch_i in np.arange(self.num_max_epochs):
            loss = None
            kfold = KFold(n_splits=10, shuffle=self.shuffle, random_state=self.random_state)

            for train_indicies, _ in kfold.split(X_train):
                self.optimizer.zero_grad()
                output = self.forward(X_train[train_indicies])
                loss = self.criterion(output, y_train[train_indicies])
                loss.backward()
                self.optimizer.step()

            # Determine if criteria for early training termination is satisfied
            if (loss_previous_epoch - loss.item()) <= self.tolerance:
                loss_consecutive_epochs_minimal = loss_consecutive_epochs_minimal + 1

                if(loss_consecutive_epochs_minimal == 50):
                    break
            else:
                loss_consecutive_epochs_minimal = 0

            loss_previous_epoch = loss.item()

    def predict(self, X_test):
        # Test the nn with the specified parameters; analogous to sklearn's
        # MLPClassifier.predict(...) method
        self.eval()
        return self.forward(X_test)

# CS598 Project Code / Logistic Regression / Implementation
class ePBRNReproducerLR(nn.Module):
    def __init__(self, num_features, inverse_reg=0.0):
        # Create the our PyTorch logistic regression model
        super(ePBRNReproducerLR, self).__init__()

        # STEP 1
        # Specify parameters for our PyTorch LR model based upon the analogous
        # parameters used by the original paper's sklearn LogisticRegression

        # PyTorch LR concept                            Analogous sklearn LogisticRegression parameter
        # ------------------                            ----------------------------------------------
        self.inverse_reg = inverse_reg                  # C (inverse of the regularization strength)
        #                                               # penalty (original paper uses L2)
        #                                               # dual (specifies dual formulation; specified but unused by the original paper)
        self.use_bias = True                            # fit_intercept (specified if bias should be added to decision function; original paper specified this as true)
        #                                               # intercept_scaling (intercept scaling, the original paper specifies this as 1)
        self.num_max_epochs = 10000                     # max_iter (maximum number of epochs when using stochastic optimizers)
        #                                               # multi_class (specifies class of problem; ours is a binary classification problem)
        #                                               # n_jobs (the number of CPU cores used for parallelization; specified but unused by the original paper)
        self.random_state = 12345                       # random_state (static, random state for reproducibility)

        # STEP 2
        # Define the layers for our lr model
        self.num_input_features = num_features

        self.fc1 = nn.Linear(in_features=self.num_input_features, out_features=1, bias=self.use_bias)

        # STEP 3
        # Define the criteria and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
            lr = 0.05,
            weight_decay=self.inverse_reg)

    def forward(self, x):
        # Perform a forward pass on the nn; it is not recommended to call this
        # function directly, but to instead call fit(...) or predict(...) so that model's
        # mode is correctly set automatically
        x = self.fc1(x)

        return torch.squeeze(x)

    def fit(self, X_train, y_train):
        # Train the LR with the specified parameters; analogous to sklearn's
        # LogisticRegression.fit(...) method
        self.train()

        loss_previous_epoch = 1.0
        loss_consecutive_epochs_minimal = 0

        for epoch_i in np.arange(self.num_max_epochs):
            loss = None
            kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)

            for train_indicies, _ in kfold.split(X_train):
                self.optimizer.zero_grad()
                output = self.forward(X_train[train_indicies])
                loss = self.criterion(output, y_train[train_indicies])
                loss.backward()
                self.optimizer.step()

            # Determine if criteria for early training termination is satisfied
            if (loss_previous_epoch - loss.item()) <= 0.0001:
                loss_consecutive_epochs_minimal = loss_consecutive_epochs_minimal + 1

                if(loss_consecutive_epochs_minimal == 50):
                    break
            else:
                loss_consecutive_epochs_minimal = 0

            loss_previous_epoch = loss.item()

    def predict(self, X_test):
        # Test the LR with the specified parameters; analogous to sklearn's
        # LogisticRegression.predict(...) method
        self.eval()
        return self.forward(X_test)
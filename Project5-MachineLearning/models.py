import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        train_ok = False
        while train_ok == False:
            for x, y in dataset.iterate_once(batch_size=1):
                #compare the target label and the actual output
                if not self.get_prediction(x) == nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    break
            else:
                train_ok = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #save the parameters W & b of each layer in lists
        self.layer_counter = 3
        self.W = [None]*self.layer_counter
        self.b = [None]*self.layer_counter

        self.layer_size = 50
        self.alpha = 0.1       #the learning rate(used in back propagation)
        self.batch_size = 100

        for layer_num in range(self.layer_counter):
            if layer_num == 0:    #the first layer
                self.W[0] = nn.Parameter(1, self.layer_size)
                self.b[0] = nn.Parameter(1, self.layer_size)
            elif layer_num == self.layer_counter - 1:    #the last layer
                self.W[layer_num] = nn.Parameter(self.layer_size, 1)
                self.b[layer_num] = nn.Parameter(1, 1)
            else:
                self.W[layer_num] = nn.Parameter(self.layer_size, self.layer_size)
                self.b[layer_num] = nn.Parameter(1, self.layer_size)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        input = x
        output = 0
        for layer_num in range(self.layer_counter):
            #use the linear and bias function in nn to update outputs
            fx = nn.Linear(input, self.W[layer_num])
            output = nn.AddBias(fx, self.b[layer_num])
            #have to print the final one
            if layer_num == self.layer_counter - 1:
                break
            #run the activate function
            else:
                input = nn.ReLU(output)  
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            #start training
            for x, y in dataset.iterate_once(self.batch_size):
                loss_target = nn.as_scalar(self.get_loss(x, y))
                grad = nn.gradients(self.get_loss(x, y), self.W + self.b)
                #use gradient descend to update the parameters
                for i in range(self.layer_counter):
                    self.W[i].update(grad[i], -self.alpha)
                    self.b[i].update(grad[len(self.W) + i], -self.alpha)
            if loss_target < 0.01:
                break
        return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #save the parameters W & b of each layer in lists
        self.layer_counter = 3
        self.W = [None]*self.layer_counter
        self.b = [None]*self.layer_counter

        self.layer_size = 200
        self.alpha = 0.1       #the learning rate(used in back propagation)
        self.batch_size = 100

        for layer_num in range(self.layer_counter):
            if layer_num == 0:    #the first layer
                self.W[0] = nn.Parameter(784, self.layer_size)
                self.b[0] = nn.Parameter(1, self.layer_size)
            elif layer_num == self.layer_counter - 1:    #the last layer
                self.W[layer_num] = nn.Parameter(self.layer_size, 10)
                self.b[layer_num] = nn.Parameter(1, 10)
            else:
                self.W[layer_num] = nn.Parameter(self.layer_size, self.layer_size)
                self.b[layer_num] = nn.Parameter(1, self.layer_size)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        input = x
        output = 0
        for layer_num in range(self.layer_counter):
            #use the linear and bias function in nn to update outputs
            fx = nn.Linear(input, self.W[layer_num])
            output = nn.AddBias(fx, self.b[layer_num])
            #have to print the final one
            if layer_num == self.layer_counter - 1:
                break
            #run the activate function
            else:
                input = nn.ReLU(output)  
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            #start training
            for x, y in dataset.iterate_once(self.batch_size):
                grad = nn.gradients(self.get_loss(x, y), self.W + self.b)
                #use gradient descend to update the parameters
                for i in range(self.layer_counter):
                    self.W[i].update(grad[i], -self.alpha)
                    self.b[i].update(grad[len(self.W) + i], -self.alpha)
            
            if dataset.get_validation_accuracy() > 0.98:
                break
        return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.alpha = 0.1
        self.batch_size = 10
        self.hidden_size = 400
        out_length = len(self.languages)

        self.W = nn.Parameter(self.num_chars, self.hidden_size)
        self.b = nn.Parameter(1,self.hidden_size)

        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden = nn.Parameter(1, self.hidden_size)

        self.W_output = nn.Parameter(self.hidden_size, out_length)
        self.b_output = nn.Parameter(1, out_length)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for x in xs:
            #the first node
            if x == xs[0]:
                hidden_state = nn.Linear(x, self.W)
            else:
                output1 = nn.AddBias(nn.Linear(x, self.W), self.b)
                output2 = nn.AddBias(nn.Linear(hidden_state, self.W_hidden), self.b_hidden)
                hidden_state = nn.ReLU(nn.Add(output1, output2))

        return nn.AddBias(nn.Linear(hidden_state, self.W_output), self.b_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                paralist = [self.W,self.b,self.W_hidden,self.b_hidden,self.W_output,self.b_output]
                gradients = nn.gradients(self.get_loss(x,y), paralist)
                for i in range(len(paralist)):
                    param = paralist[i]
                    param.update(gradients[i], -self.alpha)
            if dataset.get_validation_accuracy() > 0.88:
                break
        return

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
        weight = self.get_weights()
        return nn.DotProduct(x,weight)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        prediction = nn.as_scalar(self.run(x))
        if prediction >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        while(accuracy != 1.0):
            total = 0
            accurate = 0
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) == 1 and nn.as_scalar(y) == -1:
                    self.get_weights().update(x,nn.as_scalar(y))
                if self.get_prediction(x) == -1 and nn.as_scalar(y) == 1:
                    self.get_weights().update(x,nn.as_scalar(y))
                else:
                    accurate += 1
                total += 1
            accuracy = accurate * 1.0 / total



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = 0.005
        self.batch_size = 1
        self.layer_size = 200
        self.layer = 2
        self.w = []
        self.b = []
        for i in range(self.layer):
            if i == 0:
                self.w.append(nn.Parameter(1,self.layer_size))
                self.b.append(nn.Parameter(self.batch_size,self.layer_size))
            if i == self.layer - 1:
                self.w.append(nn.Parameter(self.layer_size,1))
                self.b.append(nn.Parameter(self.batch_size,1))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        predicted_y = x
        for i in range(self.layer):
            xm = nn.Linear(predicted_y,self.w[i])
            predicted_y = nn.AddBias(xm,self.b[i])
            if i != self.layer - 1:
                predicted_y = nn.ReLU(predicted_y)

        return predicted_y



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
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y,y)



    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_num = 1
        while loss_num > 0.02:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                loss_num = nn.as_scalar(loss)
                grad = nn.gradients(loss,self.w+self.b)
                for j in range(self.layer):
                    self.w[j].update(grad[j],-self.multiplier)
                    self.b[j].update(grad[j+len(grad)//2],-self.multiplier)



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
        self.w = []
        self.b = []
        self.batch_size = 10
        self.layer = 3
        self.layer_size = [300,100,10]
        self.multiplier = 0.002
        for i in range(self.layer):
            if i == 0:
                self.w.append(nn.Parameter(784,self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))
            else:
                self.w.append(nn.Parameter(self.layer_size[i-1],self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))




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
        predicted_y = x
        for i in range(self.layer):
            xm = nn.Linear(predicted_y,self.w[i])
            predicted_y = nn.AddBias(xm,self.b[i])
            if i != self.layer - 1:
                predicted_y = nn.ReLU(predicted_y)

        return predicted_y




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
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        loss_num = 1
        while accuracy < 0.975:
            count = 0
            count += 1
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                #loss_num = nn.as_scalar(loss)
                grad = nn.gradients(loss,self.w+self.b)
                for j in range(self.layer):
                    self.w[j].update(grad[j],-self.multiplier)
                    self.b[j].update(grad[len(grad)//2+j],-self.multiplier)
                        #print("loss ",loss_num)
            accuracy = dataset.get_validation_accuracy()
            print("accuracy ",accuracy)






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

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

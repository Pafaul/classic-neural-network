import math

class ActivationFunction:
    def calc(self, x):
        pass
    
    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def calc(self, x):
        return 1/(1+math.pow(math.e,-x))

    def derivative(self, x):
        return self.calc(x)*(1-self.calc(x))
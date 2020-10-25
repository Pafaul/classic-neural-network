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

class SigmoidParam(ActivationFunction):
    def __init__(self, k=1):
        self.k = k
    def calc(self, x):
        return 1/(1+math.pow(math.e, -self.k*x))
    
    def derivative(self, x):
        return self.k*math.pow(math.e, -self.k*x)/math.pow(1+(math.pow(math.e,-self.k*x)),2)

class LeakyReLU(ActivationFunction):
    def calc(self, x):
        if (x >= 0):
            return x
        else:
            return 0.01*x

    def derivative(self, x):
        if (x >= 0):
            return 1
        else:
            return 0.01

class SoftPlus(ActivationFunction):
    def calc(self, x):
        return ((x**2+1)**0.5-1)/2+x
    
    def derivative(self, x):
        return (x/(2*((x**2+1)**0.5))+1)

class Equal(ActivationFunction):
    def calc(self, x):
        return x

    def derivative(self, x):
        return 1

class Th(ActivationFunction):
    def calc(self, x):
        return (math.pow(math.e,x) - math.pow(math.e, -x))/(math.pow(math.e, x) + math.pow(math.e, -x))

    def derivative(self, x):
        return 1 - math.pow(self.calc(x), 2)
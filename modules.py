
import numpy as np

class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        
        # The easiest case:
            
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        
        # The easiest case:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"

class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 
         
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:
        
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   
            
            
        Just write a little loop. 
        """
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input, g_1)   
             
        
        """
        self.gradInput = gradOutput
        for i in reversed(range(1, len(self.modules))):
            self.gradInput = self.modules[i].backward(self.modules[i-1].output,self.gradInput)
        self.gradInput = self.modules[0].backward(input,self.gradInput)
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()

class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = np.add(input @ self.W.T,self.b)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput @ self.W
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = gradOutput.T @ input
        self.gradb = gradOutput.sum(axis=0)
        
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        exps = np.exp(self.output)
        self.output = np.divide(exps, np.sum(exps, axis=1, keepdims=True))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        batch_size, n_feats = self.output.shape
        self.gradInput = np.zeros_like(input)
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.subtract(np.diagflat(y), y @ y.T)
            self.gradInput[i] = gradOutput[i] @ jacobian 
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"

class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.subtract(self.output, np.log(np.sum(np.exp(self.output), axis=1, keepdims=True)))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        batch_size, n_feats = self.output.shape
        self.gradInput = np.zeros_like(input)
        for i in range(batch_size):
            S = np.exp(self.output[i]).reshape(-1, 1)
            ones = np.ones((n_feats, 1))
            jacobian = np.eye(n_feats) - (ones @ S.T)
            self.gradInput[i] = gradOutput[i] @ jacobian
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"

class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        self.x_hat = None
        self.x_centered = None
        self.inv_std = None
        self.batch_mean = None
        self.batch_var = None
        
    def updateOutput(self, input):
        self.input = input
        N, D = input.shape
        if self.moving_mean is None:
            self.moving_mean = np.zeros(D)
            self.moving_variance = np.ones(D)
            
        if self.training:
            self.batch_mean = np.mean(input, axis=0)
            self.batch_var = np.var(input, axis=0)
            
            self.moving_mean = np.add(np.multiply(self.moving_mean, self.alpha), np.multiply(self.batch_mean, np.subtract(1, self.alpha)))
            self.moving_variance = np.add(np.multiply(self.moving_variance, self.alpha), np.multiply(self.batch_var, np.subtract(1, self.alpha)))
            
            x_centered = np.subtract(input, self.batch_mean)
            self.inv_std = np.divide(1., np.sqrt(np.add(self.batch_var, self.EPS)))
            self.x_hat = np.multiply(x_centered, self.inv_std)
        else:
            x_centered = np.subtract(input, self.moving_mean)
            self.inv_std = np.divide(1., np.sqrt(np.add(self.moving_variance, self.EPS)))
            self.x_hat = np.multiply(x_centered, self.inv_std)
    
        self.x_centered = x_centered
        self.output = self.x_hat
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        N, D = self.input.shape
        inv_std = self.inv_std
        grad_x_hat = gradOutput
        term1 = np.multiply(N, grad_x_hat)
        sum_grad_x_hat = np.sum(grad_x_hat, axis=0)
        term2 = sum_grad_x_hat
        x_hat = self.x_hat
        sum_grad_x_hat_x_hat = np.sum(np.multiply(grad_x_hat, x_hat), axis=0)
        term3 = np.multiply(x_hat, sum_grad_x_hat_x_hat)
        self.gradInput = np.multiply(inv_std, np.subtract(np.subtract(term1, term2),term3))
        self.gradInput = np.multiply(1./N, self.gradInput)
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"

class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        self.input = input
    
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            self.output = input * self.mask / (1 - self.p)
        else:
            self.output = input
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask / (1 - self.p)
        else:
            self.gradInput = gradOutput
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"

class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"

class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        self.input = None
        self.output = None
        self.gradInput = None
        
    def updateOutput(self, input):
        self.input = input
        self.output = np.where(input > 0, input, self.slope * input)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        derivative = np.where(self.input > 0, 1, self.slope)
        self.gradInput = gradOutput * derivative
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"

class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        self.input = None
        self.output = None
        self.gradInput = None
        
    def updateOutput(self, input):
        self.input = input
        
        self.output = np.where(
            input > 0,
            input,
            self.alpha * (np.exp(input) - 1)
        )
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        derivative = np.where(
            self.input > 0,
            1,
            self.alpha * np.exp(self.input)
        )
        self.gradInput = gradOutput * derivative
        return self.gradInput
    
    def __repr__(self):
        return "ELU"

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
        self.input = None
        self.output = None
        self.gradInput = None
    
    def updateOutput(self, input):
        self.input = input
        self.output = np.log(1 + np.exp(input))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        sigmoid = 1.0 / (1.0 + np.exp(-self.input))
        self.gradInput = gradOutput * sigmoid
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"

class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"

class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        log_probs = np.log(input_clamp)
        self.output = -np.sum(target * log_probs) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.gradInput = -target / input_clamp / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"

class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        self.output = -np.sum(target * input) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"

def sgd_momentum(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)
            
            current_var -= old_grad
            var_index += 1     

def adam_optimizer(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * current_grad**2, out=var_second_moment)
            current_var -= lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon'])
            state['m'][var_index] = var_first_moment
            state['v'][var_index] = var_second_moment
            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1

import scipy as sp
import scipy.signal

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        batch_size, in_c, h_in, w_in = input.shape
        self.output = np.zeros((batch_size, self.out_channels, h_in, w_in))
        
        for i in range(batch_size):
            padded = np.pad(input[i], ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')
            for o in range(self.out_channels):
                channel_output = np.zeros((h_in, w_in))
                for c in range(self.in_channels):
                    corr = sp.signal.correlate(padded[c], self.W[o, c], mode='valid')
                    channel_output += corr
                channel_output += self.b[o]
                self.output[i, o] = channel_output
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        pad = self.kernel_size // 2
        batch_size, out_c, h_out, w_out = gradOutput.shape
        self.gradInput = np.zeros_like(input)
        
        for b in range(batch_size):
            grad_padded = np.zeros((self.in_channels, h_out + 2*pad, w_out + 2*pad))
            for o in range(out_c):
                for c in range(self.in_channels):
                    # Отражение ядра по высоте и ширине
                    flipped_kernel = np.flip(self.W[o, c], axis=(-2, -1))
                    corr = sp.signal.correlate(gradOutput[b, o], flipped_kernel, mode='full')
                    grad_padded[c] += corr
            # Обрезаем padding
            self.gradInput[b] = grad_padded[:, pad:-pad, pad:-pad]
        
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        pad = self.kernel_size // 2
        batch_size, in_c, h_in, w_in = input.shape
        
        for b in range(batch_size):
            # Добавляем паддинг для высоты и ширины, каналы не паддим
            padded = np.pad(input[b], ((0, 0), (pad, pad), (pad, pad)), mode='constant')
            for o in range(self.out_channels):
                for c in range(self.in_channels):
                    # Корреляция без отражения ядра
                    self.gradW[o, c] += sp.signal.correlate(padded[c], gradOutput[b, o], mode='valid')
        
        # Градиент для смещений: сумма по батчу, высоте и ширине
        self.gradb += np.sum(gradOutput, axis=(0, 2, 3))
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q

class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None
                    
    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        assert input_h % self.kernel_size == 0  
        assert input_w % self.kernel_size == 0
        
        k = self.kernel_size
        n, c, h, w = input.shape
        h_out, w_out = h // k, w // k
        
        reshaped = input.reshape(n, c, h_out, k, w_out, k)
        reshaped = reshaped.transpose(0, 1, 2, 4, 3, 5)
        reshaped_flat = reshaped.reshape(n, c, h_out, w_out, -1)
        
        self.output = reshaped_flat.max(axis=-1)
        max_indices = reshaped_flat.argmax(axis=-1)
        
        self.mask = np.zeros_like(reshaped_flat, dtype=bool)
        n_idx, c_idx, h_idx, w_idx = np.indices((n, c, h_out, w_out))
        self.mask[n_idx, c_idx, h_idx, w_idx, max_indices] = True
        
        self.mask = self.mask.reshape(reshaped.shape).transpose(0, 1, 2, 4, 3, 5)
        self.mask = self.mask.reshape(n, c, h, w)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        k = self.kernel_size
        grad_expanded = gradOutput.repeat(k, axis=2).repeat(k, axis=3)
        self.gradInput = grad_expanded * self.mask
        return self.gradInput
    
    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q

class Flatten(Module):
    def __init__(self):
         super(Flatten, self).__init__()
    
    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput
    
    def __repr__(self):
        return "Flatten"
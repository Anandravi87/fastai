#!/usr/bin/env python
# coding: utf-8

# ## Loss Function aka Training Objective

# In[2]:


#hide
get_ipython().system('pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


# In[3]:


#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')


# *Create zeros and ones tensors*

# In[4]:


path_train = Path("""/notebooks/storage/Anand/train_anand""")


# In[5]:


zeros_train = (path_train/'0').ls().sorted()


# In[6]:


ones_train = (path_train/'1').ls().sorted()


# ### list comprehension 

# In[7]:


zero_tensors_train = [tensor(Image.open(o)) for o in zeros_train]
one_tensors_train = [tensor(Image.open(o)) for o in ones_train]


# In[8]:


len(zero_tensors_train),len(one_tensors_train)


# In[8]:


stacked_zeros_train = torch.stack(zero_tensors_train).float()/255
stacked_ones_train = torch.stack(one_tensors_train).float()/255


# **View**: A Pytorch method that changes the shape of a tensor without changing its contents. 
#     -1 is a special parameter to view that means "make this axis as big as necessary to fit all the data"

# In[9]:


#initializing a new variable (instead of hard coding) to hold the number of pixels in the examples
len_ex = 28


# In[12]:


train_x = torch.cat([stacked_zeros_train, stacked_ones_train]).view(-1,len_ex*len_ex)


# Create the label. The method unsqueeze is used to covert a vector to a matrix with 1 column 

# In[15]:


train_y = tensor([1]*len(zeros_train) + [0]*len(ones_train)).unsqueeze(1)
train_x.shape,train_y.shape


# A dataset returns a tuple. A simple way to create a Dataset is to use zip() method and combine it with list()

# In[16]:


dset_train = list(zip(train_x,train_y))


# In[19]:


#verifying if the dataset (dset_train) returns tuples as expected
x,y = dset_train[2000]
x.shape,y


# Repeat the above steps to create a Validation set namely dset_valid

# In[20]:


path_valid = Path("""/notebooks/storage/Anand/test_anand""")


# In[21]:


zeros_valid = (path_valid/'0').ls().sorted()


# In[22]:


ones_valid = (path_valid/'1').ls().sorted()


# In[23]:


#List comprehension for the validation set
zero_tensors_valid = [tensor(Image.open(o)) for o in zeros_valid]
one_tensors_valid = [tensor(Image.open(o)) for o in ones_valid]


# In[24]:


len(zero_tensors_valid),len(one_tensors_valid)


# In[25]:


stacked_zeros_valid = torch.stack(zero_tensors_valid).float()/255
stacked_ones_valid = torch.stack(one_tensors_valid).float()/255


# In[26]:


stacked_zeros_valid.shape, stacked_ones_valid.shape


# In[36]:


valid_x = torch.cat([stacked_zeros_valid, stacked_ones_valid]).view(-1,len_ex*len_ex)
valid_y = tensor([1]*len(zeros_valid) + [0]*len(ones_valid)).unsqueeze(1)
valid_x.shape,valid_y.shape


# In[47]:


dset_valid = list(zip(valid_x,valid_y))


# # Summary of Gradient Descent
# 1. Initialize the weights.
# 2. For each image, use these weights to predict whether it appears to be a 0 or a 1.
# 3. Based on these predictions, calculate how good the model is (its loss).
# 4. Calculate the gradient, which measures for each weight, how changing that weight would change the loss
# 5. Step (that is, change) all the weights based on that calculation.
# 6. Go back to the step 2, and repeat the process.
# 7. Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).

# # The loss function

# In[48]:


#initialize weights randomly (step 1 of GD)
def init_params(size, var=1.0): 
    return (torch.randn(size)*var).requires_grad_()


# In[49]:


#randomly initialising weights and biases
weights = init_params((len_ex*len_ex,1))
bias = init_params(1)


# In[50]:


#defining a function to perform matrix multiplication. The input is multiplied with the weights and the bias is added
def matrix_mult(ex): 
    return ex@weights + bias


# In[51]:


#defining the loss function. Passing the preds through sigmoid because the loss function assumes that the 
#preds are always between 0 and 1

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


# In[52]:


preds = matrix_mult(train_x)


# In[53]:


corrects = (preds>0.0).float() == train_y


# In[54]:


corrects.float().mean().item()


# # Optimisation step
# Updating the weights based on gradient calculations
# We calculate the average loss for a few data items at a time. This is called a mini-batch. 
# 
# Batch size: the number of data items in the mini-batch
# 
# DataLoader is a class that does the shuffling and mini-batch collation. It can take any Python collection and turn it into an iterator over many batches
# 
# torch.randn() --> generate random numbers from the standard normal distribution
# 
# Dataset: A collection that contains tuples of independent and dependent variables
# 
# Inplace Operations: Methods in PyTorch whose names end in an underscore modify their objects in place. For instance, bias.zero_() sets all elements of the tensor bias to 0.

# In[56]:


#randomly initialising weights and biases
weights = init_params((len_ex*len_ex,1))
bias = init_params(1)


# In[58]:


#creating a dataloader class object
#Experimenting with a batch size of 50 (instead of 256)

# for training
train_dl = DataLoader(dset_train, batch_size=50)

#for validation
valid_dl = DataLoader(dset_valid, batch_size=50)


# In[60]:


#Define a function to compute gradient

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


# In[61]:


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()


# In[62]:


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


# # Creating an optimiser
# 
# The main objective is to refactor the code in the previous sections into simpler and easy to debug code
# 
# We will use PyTorch's nn.Linear module. A module is an object of a class that inherits from the PyTorch nn.Module class.

# In[63]:


linear_model = nn.Linear(len_ex*len_ex,1)


# In[64]:


#Retrieving weights and bias from the result of calling nn.Linear

w,b = linear_model.parameters()

# Initialising some important variables

lr = 1.
params = weights,bias


# In[69]:


#Creating a Optimizer class - basically, this initializes the params, steps them and resets them to 0 after the step)

class BasicOptim:
    def __init__(self,params,lr): 
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None


# In[70]:


opt = BasicOptim(linear_model.parameters(), lr)


# In[66]:


#define train_epoch using the Optimizer class

def train_epoch(model):
    for xb,yb in train_dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


# In[71]:


#define a function for the training loop

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')


# In[72]:


train_model(linear_model, 20)


# In[73]:


linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 50)


# In[74]:


dls = DataLoaders(train_dl, valid_dl)


# In[75]:


learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)


# In[ ]:


#def mnist_loss(predictions, targets):
#    predictions = predictions.sigmoid()
#    return torch.where(targets==1, 1-predictions, predictions).mean()


# In[ ]:


#def batch_accuracy(xb, yb):
#    preds = xb.sigmoid()
#    correct = (preds>0.5) == yb
#    return correct.float().mean()


# In[76]:


learn.fit(30, lr=lr)


# In[ ]:





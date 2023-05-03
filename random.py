"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains useful info about random sampling with different python packages.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import numpy as np
import torch
import tensorflow as tf

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------
'''
You'll see here that the random module is totally useless! Numpy provides everything that random provides, but much 
better!
Reference:
https://docs.python.org/3/library/random.html
'''

# Getting random numbers (uniform distribution) between 0 and 1:

r = np.random.rand(10, 2)          # Using Numpy, we can get matrices of random numbers.
print(r)
'''
r = np.random.rand()                # gives 1 random number between 0 and 1
r = np.random.random(size=(10, 2))  # Same as above (worse!)
r = random.random()                 # same as above (worse!)
'''



# Getting random integers:

r = np.random.randint(low=20, high=80, size=(10, 2))                # with replacement and we can't change it :|
print(r)
r = np.random.choice(range(20, 81), size=(10, 2), replace=False)    # This is like above but without replacement
print(r)
'''
r = random.randrange(101)                                           # integer from 0 to 101 inclusive
r = random.randrange(start=0, stop=100, step=2)                     # integer from 0 100 inclusive
'''


# Sampling a list:

mylist = ['apple', 'banana', 'cherry', 'peach', 'strawberry', 'nectarine', 'orange']

r = np.random.choice(mylist, size=20, replace=True)     # here we have the option of sampling with replacement
print(r)
'''
r = random.sample(mylist, k=4)                          # here we don't have the option of sampling with replacement
r = random.choice(mylist)                               # single random element from a list
'''

# Shuffling a list:
mylist = ['apple', 'banana', 'cherry', 'peach', 'strawberry', 'nectarine', 'orange']
np.random.shuffle(mylist)
'''
random.shuffle(mylist)          # random module equivalent
'''

# Sampling non-uniform distributions:
r = np.random.uniform(low=0, high=20, size=10)
print(r)
r = np.random.normal(loc=0, scale=1, size=10)
print(r)
r = torch.randn(3, 4)            # return a 3 x 4 tensor with random numbers ~ normal distribution
print(r)

r = np.random.exponential(scale=1, size=10)
print(r)
r = np.random.beta(a=.1, b=.2, size=10)
print(r)

# Binomial/Bernoulli distribution (flipping a coin)
r = np.random.binomial(n=1, p=.5, size=1000)          # same as the line below: Binomial with n=1 is Bernouli
print(r)
# r = bernoulli.rvs(.5, size=1000)                      # rvs: random variables

# Multinomial distribution (rolling a dice)
r = np.random.multinomial(n=1, pvals=[.3, .2, .5], size=1000)       # same as the line below
print(r)
# r = multinomial.rvs(p=[.3, .2, .5], n=1, size=1000)

'''
The random module equivalents are useless, since they don't give the chance to draw multiple samples at once:
r = random.uniform(2.5, 10)
r = random.gauss(mu=2, sigma=1)
r = random.expovariate(lambd=.1)
r = random.betavariate(alpha=0.2, beta=0.1)
'''


# -------------------------- PyTorch random functions --------------------------

# Getting random numbers (uniform distribution) between 0 and 1:
r = torch.rand(10, 2)     # torch.rand((10, 2)) also works: tensor shaped [10, 2] with uniform distribution over [0, 1)
print(r)

# Getting random numbers (uniform distribution) between 0 and 1, shaped :
lst = [[1, 2], [3, 4]]
tnsr = torch.tensor(lst)
rand_tnsr = torch.rand_like(tnsr, dtype=torch.float)    # overrides the datatype of x_data
print(f"Random Tensor: \n {rand_tnsr} \n")


# -------------------------- TensorFlow random functions --------------------------

r = tf.random.normal(shape=(2, 3), mean=0, stddev=1.5)
print(r)
r = tf.random.uniform(shape=(2, 3), minval=0, maxval=10, dtype='int32')
print(r)


# -------------------------------------------------- CODE TESTING -----------------------------------------------------

img_shape = np.array((500, 600, 700))
patch_shape = np.array((128, 128, 128))
diff = img_shape - patch_shape
patch_origin = np.random.randint(low=0, high=diff+1)
patching_slicer = []
for dim in range(len(patch_origin)):
    patching_slicer.append(slice(patch_origin[dim], patch_origin[dim] + patch_shape[dim]))
# patching_slicer = tuple(patching_slicer)

img = np.random.rand(500, 600, 700)
img2 = img[patching_slicer]
img2.shape
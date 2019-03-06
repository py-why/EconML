# Deep IV in ALICE

We're providing a suite of tools called ALICE for estimating conditional average treatment effects (CATEs).  Since our library is in development, I've pulled out our Deep IV estimator (`DeepIvEstimator`, in [deepiv.py](./deepiv.py)) as well as the base class that it specializes (`BaseCateEstimator`, in [cate_estimator.py](./cate_estimator.py)) and put them into this zip file.

## Use

Our Deep IV estimator is built on top of Keras; we support either the Tensorflow or the Theano backends.  There are three steps to using the `DeepIvEstimator`:

1. Construct an instance.  
    * The `m` and `h` arguments to the initializer are functions that each take two Keras inputs and return a Keras model (the inputs are `z` and `x` in the case of `m`; the inputs are `t` and `x` in the case of `h` and the output's shape should match `y`).  Note that the `h` function will be called multiple times, but should reuse the same weights - see below for a concrete example of how to get this right using the Keras API.
    * The `n_samples`, `use_upper_bound_loss`, and `n_gradient_samples` arguments together determine how the loss for the response model will be computed.
        * If `use_upper_bound_loss` is `False` and `n_gradient_samples` is zero, then `n_samples` samples will be averaged to approximate the response - this will provide an unbiased estimate of the correct loss only in the limit as the number of samples goes to infinity.
        * If `use_upper_bound_loss` is `False` and `n_gradient_samples` is nonzero, then we will average `n_samples` samples to approximate the response a first time and average `n_gradient_samples` samples to approximate it a second time - combining these allows us to provide an unbiased estimate of the true loss.
        * If `use_upper_bound_loss` is `True`, then `n_gradient_samples` must be `0`; `n_samples` samples will be used to get an unbiased estimate of an upper bound of the true loss - this is equivalent to adding a regularization term penalizing the variance of the response model
2. Call `fit`; this will train both models.  Note that the `W` argument must always be `None` - it is only here to maintain API consistency with other methods that do support controls separate from features.
3. Call `effect` or `predict` depending on what output you want.  Effect calculates the difference in outcomes based on the features and the two different treatments, while predict predicts the outcome based on a single treatment.

### Example

```python

d_t = 2 # number of treatments
d_z = 1 # number of instruments
d_x = 3 # number of features
d_y = 2 # number of responses
n = 5000
# simple DGP only for illustration
x = np.random.uniform(size=(n,d_x))
z = np.random.uniform(size=(n,d_z))
p_x_t = np.random.uniform(size=(d_x,d_t))
p_z_t = np.random.uniform(size=(d_z,d_t))
t = x @ p_x_t_ + z @ p_z_t
p_xt_y = np.random.uniform(size=(d_x*d_t,d_y))
y =  (x.reshape(n,-1,1) * t.reshape(n,1,-1)).reshape(n,-1) @ p_xt_y

# Define the treatment model neural network architecture
# This will take the concatenation of one-dimensional values z and x as input, so the input shape is (d_z + d_x,)
# The exact shape of the final layer is not critical because the Deep IV framework will add extra layers on top for the mixture density network
treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(d_z + d_x,)),
                                    keras.layers.Dropout(0.17),
                                    keras.layers.Dense(64, activation='relu'),
                                    keras.layers.Dropout(0.17),
                                    keras.layers.Dense(32, activation='relu'),
                                    keras.layers.Dropout(0.17)])

# Define the response model neural network architecture
# This will take the concatenation of one-dimensional values t and x as input, so the input shape is (d_t + d_x,)
# The output should match the shape of y, so it must have shape (d_y,) in this case
# NOTE: For the response model, it is important to define the model *outside* of the lambda passed to the DeepIvEstimator, as we do here,
#       so that the same weights will be reused in each instantiation
response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(d_t + d_x,)),
                                   keras.layers.Dropout(0.17),
                                   keras.layers.Dense(64, activation='relu'),
                                   keras.layers.Dropout(0.17),
                                   keras.layers.Dense(32, activation='relu'),
                                   keras.layers.Dropout(0.17),
                                   keras.layers.Dense(d_y)])

deepIv = DeepIVEstimator(n_components = 10, # number of gaussians in our mixture density network
                         m = lambda z, x : treatment_model(keras.layers.concatenate([z,x])), # treatment model
                         h = lambda t, x : response_model(keras.layers.concatenate([t,x])),  # response model
                         n_samples = 1, # number of samples to use to estimate the response
                         use_upper_bound_loss = False, # whether to use an approximation to the true loss
                         n_gradient_samples = 1, # number of samples to use in second estimate of the response (to make loss estimate unbiased)
                         optimizer='adam', # Keras optimizer to use for training - see https://keras.io/optimizers/ 
                         s1=100, # number of epochs to train treatment model
                         s2=100) # number of epochs to train response model

deepIv.fit(Y=y,T=t,X=x,W=None,Z=z)
# do something with predictions...
deepIv.predict(T=t,X=x)

```

### Notes

Due to the way that we build our models, it is normal to see warnings indicating that outputs are missing from the loss dictionary; these can be ignored.
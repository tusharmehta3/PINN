from tensorflow.python.client import device_lib

device_lib.list_local_devices()
import os
import numpy as np
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import griddata
import pandas as pd


# Example data
data = scipy.io.loadmat('output.mat')
P_star = data['p']
U_star = data['v']  # Replace with your velocity data
t_star = data['t']  # Replace with your time data
X_star = data['points']
# Get the indices that would sort the time array
sorted_indices = np.argsort(t_star[0])

# Use the indices to sort the time array
sorted_time = t_star[:, sorted_indices]

# Use the indices to sort the last dimension of the velocity array
sorted_velocity = U_star[:, :, sorted_indices]
sorted_pressure = P_star[:, sorted_indices]
sorted_points = X_star[:, :, sorted_indices]

print("Sorted time shape:", sorted_time.shape)
print("Sorted velocity shape:", sorted_velocity.shape)

U_star = sorted_velocity[:,:,:250]
P_star = sorted_pressure[:,:250]
X_star = sorted_points[:,:,:250]
t_star = sorted_time[:,:250]



# print("First 10 elements of sorted time:", sorted_time)
# print("First 10 elements of velocity for the first sample and first component:",
#       sorted_velocity[0, 0, :10])

# import numpy as np


# Function to load and prepare data
def load_data():
    data = scipy.io.loadmat('output.mat')
    U_star = data['v']
    P_star = data['p']
    X_star = data['points']
    t_star = np.arange(1, 401)[:, None]

    N = X_star.shape[0]
    T = X_star.shape[2]
    XX = np.tile(X_star[:, 0, :], (1, 1))
    YY = np.tile(X_star[:, 1, :], (1, 1))
    TT = np.tile(t_star.T, (N, 1))
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = U_star[:, 0, :].flatten()[:, None]
    v = U_star[:, 1, :].flatten()[:, None]
    p = P_star.flatten()[:, None]

    return x, y, t, u, v, p

# Define Vanilla Neural Network model
class VanillaNN(tf.keras.Model):
    def __init__(self, layers):
        super(VanillaNN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='swish') for layer in layers[:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1])

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def train_vanilla_nn(x_train, y_train, t_train, u_train, v_train, p_train, layers, epochs=600, batch_size=500):
    inputs = np.concatenate([x_train, y_train, t_train], axis=1)
    outputs = np.concatenate([u_train, v_train, p_train], axis=1)

    model = VanillaNN(layers)
    model.compile(optimizer='nadam', loss='mse', metrics = ['acc'])
    model.optimizer.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=5000,
            decay_rate=0.90,
            staircase=False
        )
    model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

def predict_vanilla_nn(model, x_star, y_star, t_star):
    inputs = np.concatenate([x_star, y_star, t_star], axis=1)
    predictions = model.predict(inputs)
    u_pred = predictions[:, 0:1]
    v_pred = predictions[:, 1:2]
    p_pred = predictions[:, 2:3]
    return u_pred, v_pred, p_pred

# Load data
# x, y, t, u, v, p = load_data()

# # Split data into training and validation sets
# N_train = 20000
# N_val = 4000
# idx = np.random.choice(len(x), N_train + N_val, replace=False)
# x_train, y_train, t_train = x[idx[:N_train]], y[idx[:N_train]], t[idx[:N_train]]
# u_train, v_train, p_train = u[idx[:N_train]], v[idx[:N_train]], p[idx[:N_train]]
# x_val, y_val, t_val = x[idx[N_train:]], y[idx[N_train:]], t[idx[N_train:]]
# u_val, v_val, p_val = u[idx[N_train:]], v[idx[N_train:]], p[idx[N_train:]]
# import numpy as np
# import tensorflow as tf
# import scipy.io
# import os

# Define the region of interest (ROI)
x_min, x_max = 3, 16
y_min, y_max = -6, 6
tolerance = 1e-1

def filter_data_within_roi(X_star, U_star, P_star, t_star, x_min, x_max, y_min, y_max):
    mask = np.logical_and.reduce((X_star[:, 0,:] >= x_min, X_star[:, 0,:] <= x_max, 
                                  X_star[:, 1,:] >= y_min, X_star[:, 1,:] <= y_max))
    # print(mask.shape)
    X_star_filtered = X_star[:,0,:][mask]
    # print(X_star_filtered.shape)
    Y_star_filtered = X_star[:,1,:][mask]
    u_filtered = U_star[:,0,:][mask]
    v_filtered = U_star[:,1,:][mask]
    P_star_filtered = P_star[mask]
    # print(mask.shape, X_star_filtered.shape[0]/t_star.shape[1])
    t_filtered = np.tile(t_star, (X_star_filtered.shape[0]//t_star.shape[1], 1)).flatten()
    # print(t_filtered.shape)
    # u_filtered = U_star_filtered[:, 0, :].flatten()[:, None]
    # v_filtered = U_star_filtered[:, 1, :].flatten()[:, None]
    p_filtered = P_star_filtered.flatten()
    
    return X_star_filtered.flatten(), Y_star_filtered.flatten(), t_filtered, u_filtered, v_filtered, p_filtered
def filter_boundary_data(x, y, t, u, v, p, x_min, x_max, y_min, y_max):
    mask = ((np.abs(x - x_min) < tolerance) | (np.abs(x - x_max) < tolerance) | 
            (np.abs(y - y_min) < tolerance) | (np.abs(y - y_max) < tolerance))
    return x[mask], y[mask], t[mask], u[mask], v[mask], p[mask]
    
def visualize_at_timestamp(x, y, t, u, timestamp):
    # Filter data for the given timestamp
    mask = (t == timestamp)
    x_t = x[mask]
    y_t = y[mask]
    u_t = u[mask]

    # Plot the actual velocities
    plt.figure(figsize=(6, 6))
    plt.scatter(x_t, y_t, c=u_t, cmap='viridis')
    plt.colorbar(label='U velocity')
    plt.title('U velocity at t={}'.format(timestamp))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



# Load data
# data = scipy.io.loadmat('/path/to/your/output.mat')  # Update this path
# U_star = data['v']
# P_star = data['p']
# X_star = data['points']
# t_star = np.arange(1, 401)[:, None]
# U_star = sorted_velocity[:,:,:250]
# P_star = sorted_pressure[:,:250]
# X_star = sorted_points[:,:,:250]
# t_star = sorted_time[:,:250]
# Filter data within the ROI
x_roi, y_roi, t_roi, u_roi, v_roi, p_roi = filter_data_within_roi(X_star, U_star, P_star, t_star, x_min, x_max, y_min, y_max)
# print(x_sr.shape)
N = X_star.shape[0]
T = X_star.shape[2]
XX = np.tile(X_star[:, 0, :], (1, 1))
YY = np.tile(X_star[:, 1, :], (1, 1))
TT = np.tile(t_star, (N, 1)).T
x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = U_star[:, 0, :].flatten()[:, None]
v = U_star[:, 1, :].flatten()[:, None]
p = P_star.flatten()[:, None]


# print(x_roi.shape)
# Split data into training and validation sets
# Define the timestamp you want to visualize
# print(len(np.unique(t_roi)))


def visualize_at_timestamp(x, y, t, u, timestamp, index):
    # Filter data for the given timestamp
    mask = (t == timestamp)
    x_t = x[mask]
    y_t = y[mask]
    u_t = u[mask]
    print(u_t.min())

    # Define the region for plotting
    lb = np.array([x_t.min(), y_t.min()])
    ub = np.array([x_t.max(), y_t.max()])
    nn = 200
    x_grid = np.linspace(lb[0], ub[0], nn)
    y_grid = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate the data
    U_star = griddata((x_t.flatten(), y_t.flatten()), u_t.flatten(), (X, Y), method='cubic')

    # Plot the interpolated data
    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.title('U velocity at t={}'.format(timestamp))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # print(x_t, y_t)


# Define the timestamp you want to visualize
# timestamp_to_visualize = 160  # Update this with your desired timestamp

# Visualize the data at the specified timestamp
# visualize_at_timestamp(x_roi, y_roi, t_roi, u_roi, timestamp_to_visualize, index=1)
# import numpy as np

def select_fixed_data_points(x, y, t, u, v, p, points_per_timestamp=80):
    unique_timestamps = np.unique(t)
    x_selected, y_selected, t_selected, u_selected, v_selected, p_selected = [], [], [], [], [], []
    
    for timestamp in unique_timestamps:
        mask = (t == timestamp)
        x_t = x[mask]
        y_t = y[mask]
        t_t = t[mask]
        u_t = u[mask]
        v_t = v[mask]
        p_t = p[mask]
        
        if x_t.shape[0] > points_per_timestamp:
            indices = np.random.choice(x_t.shape[0], points_per_timestamp, replace=False)
            x_t, y_t, t_t, u_t, v_t, p_t = x_t[indices], y_t[indices], t_t[indices], u_t[indices], v_t[indices], p_t[indices]
        
        x_selected.append(x_t)
        y_selected.append(y_t)
        t_selected.append(t_t)
        u_selected.append(u_t)
        v_selected.append(v_t)
        p_selected.append(p_t)
    
    x_selected = np.vstack(x_selected)
    y_selected = np.vstack(y_selected)
    t_selected = np.vstack(t_selected)
    u_selected = np.vstack(u_selected)
    v_selected = np.vstack(v_selected)
    p_selected = np.vstack(p_selected)
    
    return x_selected, y_selected, t_selected, u_selected, v_selected, p_selected

# Example usage:
x_train, y_train, t_train, u_train, v_train, p_train = select_fixed_data_points(x_roi, y_roi, t_roi, u_roi, v_roi, p_roi, points_per_timestamp=120)


N_train = x_train.shape[0]
N_val = N_train // 5
idx = np.random.choice(len(x_roi), N_train + N_val, replace=False)
# x_train, y_train, t_train = x_roi[idx[:N_train]], y_roi[idx[:N_train]], t_roi[idx[:N_train]]
# u_train, v_train, p_train = u_roi[idx[:N_train]], v_roi[idx[:N_train]], p_roi[idx[:N_train]]
x_val, y_val, t_val = x_roi[idx[N_train:]], y_roi[idx[N_train:]], t_roi[idx[N_train:]]
u_val, v_val, p_val = u_roi[idx[N_train:]], v_roi[idx[N_train:]], p_roi[idx[N_train:]]
# print(x_train.shape, t_train.shape)
# Define your model (ensure the PhysicsInformedNN class is defined as before)
x_fig, y_fig, t_fig, u_fig = x_train, y_train, t_train, u_train
# layers = [3, 20, 20, 20, 20, 20, 20, 3]
# characteristic_length = 2.0  # Set this to the characteristic length of your problem
# visualize_at_timestamp(x_fig, y_fig, t_fig, u_fig, timestamp_to_visualize, index=1)
# Initial data (ensure this data is loaded correctly)
initial_data = scipy.io.loadmat('inct.mat')  # Update this path
X_star_i = initial_data['points']
U_star_i = initial_data['v']
P_star_i = initial_data['p']
t_star_i = initial_data['t']  # Assuming it contains t=0
def filter_initial_conditions_within_roi(X_star, U_star, P_star, x_min, x_max, y_min, y_max):
    # mask = np.logical_and.reduce((X_star[:, 0] >= x_min, X_star[:, 0] <= x_max, 
    #                               X_star[:, 1] >= y_min, X_star[:, 1] <= y_max))
    # X_star_filtered = X_star[mask]
    # U_star_filtered = U_star[mask]
    # P_star_filtered = P_star[mask]
    
    # u_filtered = U_star_filtered[:, 0].flatten()[:, None]
    # v_filtered = U_star_filtered[:, 1].flatten()[:, None]
    # p_filtered = P_star_filtered.flatten()[:, None]

    mask = np.logical_and.reduce((X_star[:, 0,:] >= x_min, X_star[:, 0,:] <= x_max, 
                                  X_star[:, 1,:] >= y_min, X_star[:, 1,:] <= y_max))
    X_star_filtered = X_star[:,0,:][mask]
    # print(X_star_filtered.shape)
    Y_star_filtered = X_star[:,1,:][mask]
    u_filtered = U_star[:,0,:][mask]
    v_filtered = U_star[:,1,:][mask]
    p_filtered = P_star[mask].flatten()
    
    return X_star_filtered.flatten(), Y_star_filtered.flatten(), u_filtered, v_filtered, p_filtered

x_i, y_i, u_i, v_i, p_i = filter_initial_conditions_within_roi(X_star_i, U_star_i, P_star_i, x_min, x_max, y_min, y_max)

# Filter boundary condition points

# Extract initial condition points from inct.mat
# x_i = X_star_i[:, 0:1].flatten()[:, None]
# y_i = X_star_i[:, 1:2].flatten()[:, None]
t_i = np.zeros_like(x_i)  # t=0 for initial condition
# u_i = U_star_i[:, 0, :].flatten()[:, None]
# v_i = U_star_i[:, 1, :].flatten()[:, None]
# p_i = P_star_i.flatten()[:, None]
# print(t_i.shape, t_star_i)
# visualize_at_timestamp(X_star_i[:,0,:].flatten(),X_star_i[:,1,:].flatten() , t_i, U_star_i[:,0,:].flatten(), 0, index=1)

# Filter initial conditions within the ROI
# x_i, y_i, t_i, u_i, v_i, p_i = filter_data_within_roi(x_i, y_i, t_i, u_i, v_i, p_i, x_min, x_max, y_min, y_max)

# Filter boundary condition points
x_b, y_b, t_b, u_b, v_b, p_b = filter_boundary_data(x_roi, y_roi, t_roi, u_roi, v_roi, p_roi, x_min, x_max, y_min, y_max)
# print(x_i.shape)
# print(np.unique(y_b))
# Sample boundary condition points (to reduce the number of points)
idx_rb = np.random.choice(x_b.shape[0], 10000, replace=False)
x_b = x_b[idx_rb]
y_b = y_b[idx_rb]
t_b = t_b[idx_rb]
u_b = u_b[idx_rb]
v_b = v_b[idx_rb]
p_b = p_b[idx_rb]

# Sample initial condition points (to reduce the number of points)
idx_ri = np.random.choice(x_i.shape[0], 120, replace=False)
x_i = x_i[idx_ri]
y_i = y_i[idx_ri]
t_i = t_i[idx_ri]
u_i = u_i[idx_ri]
v_i = v_i[idx_ri]
p_i = p_i[idx_ri]

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
t_train = t_train.reshape(-1, 1)
u_train = u_train.reshape(-1, 1)
v_train = v_train.reshape(-1, 1)
p_train = p_train.reshape(-1, 1)
x_val = x_val.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
t_val = t_val.reshape(-1, 1)
u_val = u_val.reshape(-1, 1)
v_val = v_val.reshape(-1, 1)
p_val = p_val.reshape(-1, 1)
x_b = x_b.reshape(-1, 1)
y_b = y_b.reshape(-1, 1)
t_b = t_b.reshape(-1, 1)
u_b = u_b.reshape(-1, 1)
v_b = v_b.reshape(-1, 1)
p_b = p_b.reshape(-1, 1)
x_i = x_i.reshape(-1, 1)
y_i = y_i.reshape(-1, 1)
t_i = t_i.reshape(-1, 1)
u_i = u_i.reshape(-1, 1)
v_i = v_i.reshape(-1, 1)
p_i = p_i.reshape(-1, 1)
def calculate_strouhal_number_at_specific_time(u_star, t_star, characteristic_length, target_time):
    """
    Calculate the Strouhal number for a specific time value.
    
    Parameters:
    - u_star: The predicted velocity data (2D array, where rows are points and columns are time steps)
    - t_star: The array of time values
    - characteristic_length: Characteristic length for Strouhal number calculation
    - target_time: The specific time step to calculate Strouhal number (e.g., 230)
    
    Returns:
    - Strouhal number at the target time step
    """
    
    # Find the index corresponding to the target time
    target_idx = np.where(t_star == target_time)[0]
    # Extract the velocity data at the target time
    u_star_at_target_time = u_star[target_idx]
    print(u_star_at_target_time.max())
    # Reshape the data to have time on the second axis for FFT
    u_star_reshaped = u_star_at_target_time.reshape(-1, 1)
    
    # Calculate the mean velocity for normalization
    mean_velocity = np.mean(u_star_reshaped, axis=0, keepdims=True)
    u_normalized = u_star_reshaped - mean_velocity
    print(u_normalized.max())
    # Perform FFT to find the dominant frequency at the target time step
    fft_result = np.fft.fft(u_star_reshaped, axis=0)
    fft_frequencies = np.fft.fftfreq(u_star_reshaped.shape[0], d=t_star[1] - t_star[0])
    print(np.argmax(np.abs(fft_result[:]), axis=0))
    dominant_frequency_idx = np.argmax(np.abs(fft_result[:]), axis=0)
    dominant_frequency = np.abs(fft_frequencies[dominant_frequency_idx])
    print(dominant_frequency_idx)
    # Calculate Strouhal number
    strouhal_number = dominant_frequency * characteristic_length / np.abs(mean_velocity).max()
    
    return strouhal_number


print(calculate_strouhal_number_at_specific_time(v_roi, t_roi, 2.0, 244.0))
# Define layers for the vanilla neural
# print(len(v_roi))
#  network
lyers = [3, 32,32,32,32,32,32,32,32,32,32, 3]
# print(np.unique(x_b))
# Train Vanilla Neural Network
# vanilla_nn = train_vanilla_nn(x_train, y_train, t_train, u_train, v_train, p_train, lyers, epochs=120)
# cylinder = 0m/s
# don't make something which is not important, make it easy, don't complicate, focus on the main part
#probelms and benefits of VNN(NO IC BC but also no result)
# if no outcome then why

# Extract weights and biases from the trained VanillaNN
vanilla_nn_weights = []
vanilla_nn_biases = []

# for layer in vanilla_nn.hidden_layers:
#     vanilla_nn_weights.append(layer.get_weights()[0])
#     vanilla_nn_biases.append(layer.get_weights()[1])

# # Append the output layer weights and biases
# vanilla_nn_weights.append(vanilla_nn.output_layer.get_weights()[0])
# vanilla_nn_biases.append(vanilla_nn.output_layer.get_weights()[1])


# import numpy as np
# import tensorflow as tf

# from scipy.interpolate import griddata

# import matplotlib.pyplot as plt

# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)



np.random.seed(1234)
tf.random.set_seed(1234)
tf.config.optimizer.set_jit(True)
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()  # Register the class for serialization
class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, x=None, y=None, t=None, u=None, v=None, layers=None, 
                 x_b=None, y_b=None, t_b=None, u_b=None, v_b=None, p_b=None, 
                 x_i=None, y_i=None, t_i=None, u_i=None, v_i=None, p_i=None, 
                 characteristic_length=None, initial_weights=None, initial_biases=None, **kwargs):
        super(PhysicsInformedNN, self).__init__(**kwargs)  # Pass all kwargs to the parent class
        X = np.concatenate([x, y, t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]
        self.u = u
        self.v = v
        self.layers_config = layers
        self.characteristic_length = characteristic_length
        self.data = {
            'iterations': [],
            'loss': [],
            'accuracy': [],
            'lambda_1': [],
            'lambda_2': [],
            'strouhal_number': []
        }

        # Boundary and initial conditions
        self.x_b, self.y_b, self.t_b = x_b, y_b, t_b
        self.u_b, self.v_b, self.p_b = u_b, v_b, p_b
        self.x_i, self.y_i, self.t_i = x_i, y_i, t_i
        self.u_i, self.v_i, self.p_i = u_i, v_i, p_i

        if initial_weights is not None and initial_biases is not None:
            self.nn_weights = [tf.Variable(w, dtype=tf.float32) for w in initial_weights]
            self.nn_biases = [tf.Variable(b, dtype=tf.float32) for b in initial_biases]
        else:
            self.nn_weights, self.nn_biases = self.initialize_NN(layers)
        
        self.lambda_1 = tf.Variable(0.0, dtype=tf.float32)
        self.lambda_2 = tf.Variable(0.0, dtype=tf.float32)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=18000,
            decay_rate=0.90,
            staircase=True
        ))

    def get_config(self):
        config = super(PhysicsInformedNN, self).get_config()
        config.update({
            'x': self.x,
            'y': self.y,
            't': self.t,
            'u': self.u,
            'v': self.v,
            'layers': self.layers_config,
            'x_b': self.x_b,
            'y_b': self.y_b,
            't_b': self.t_b,
            'u_b': self.u_b,
            'v_b': self.v_b,
            'p_b': self.p_b,
            'x_i': self.x_i,
            'y_i': self.y_i,
            't_i': self.t_i,
            'u_i': self.u_i,
            'v_i': self.v_i,
            'p_i': self.p_i,
            'characteristic_length': self.characteristic_length,
            'initial_weights': [w.numpy() for w in self.nn_weights],
            'initial_biases': [b.numpy() for b in self.nn_biases],
        })
        return config

    @classmethod
    def from_config(cls, config):
        initial_weights = config.pop('initial_weights', None)
        initial_biases = config.pop('initial_biases', None)
        instance = cls(**config)
        if initial_weights is not None and initial_biases is not None:
            instance.nn_weights = [tf.Variable(w, dtype=tf.float32) for w in initial_weights]
            instance.nn_biases = [tf.Variable(b, dtype=tf.float32) for b in initial_biases]
        return instance
            
    def initialize_NN(self, layers):
        nn_weights = []
        nn_biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            nn_weights.append(W)
            nn_biases.append(b)
        return nn_weights, nn_biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, nn_weights, nn_biases):
        num_layers = len(nn_weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = tf.cast(nn_weights[l], tf.float32)
            b = tf.cast(nn_biases[l], tf.float32)
            H = tf.cast(H, tf.float32)
            H = tf.nn.swish(tf.add(tf.matmul(H, W), b))
        W = tf.cast(nn_weights[-1], tf.float32)
        b = tf.cast(nn_biases[-1], tf.float32)
        H = tf.cast(H, tf.float32)
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        t = tf.cast(t, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            
            u_v_p = self.neural_net(tf.concat([x, y, t], 1), self.nn_weights, self.nn_biases)
            u = u_v_p[:, 0:1]
            v = u_v_p[:, 1:2]
            p = u_v_p[:, 2:3]
            # psi = psi_p[:,0:1]
            # p = psi_p[:,1:2]
            # print(tf.concat([x, y, t], 1))
            # u = tape.gradient(psi, y)
            # v = -tape.gradient(psi, x)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
        u_t = tape.gradient(u, t)
        v_t = tape.gradient(v, t)
        
        if u is None or v is None:
            raise ValueError("Gradients for u or v are None.")
        
        if u_x is None or u_y is None or v_x is None or v_y is None or u_t is None or v_t is None:
            raise ValueError("Gradients for u_x, u_y, v_x, or v_y are None.")
        
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        
        if u_xx is None or u_yy is None or v_xx is None or v_yy is None:
            raise ValueError("Second-order gradients for u_xx, u_yy, v_xx, or v_yy are None.")
        
        f_u =  lambda_1*(u_t +u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy) # use scientific(physics) terms to explain, advection convection(material derivative)
        f_v =  lambda_1*(v_t +u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
        # f_u = (u_t +  u * u_x + v * u_y) + p_x /lambda_1 - lambda_2 * (u_xx + u_yy)
        # f_v = (v_t +  u * v_x + v * v_y) + p_y /lambda_1 - lambda_2 * (v_xx + v_yy)
        cont = u_x + v_y
        del tape
        return u, v, p, f_u, f_v, cont
        # lambda_1 = self.lambda_1  # Inverse of Reynolds number
        # lambda_2 = self.lambda_2  # Related to viscosity
        # x, y, t = map(tf.cast, [x, y, t], [tf.float32] * 3)
        
        # with tf.GradientTape(persistent=True) as tape:
        #     tape.watch([x, y, t])
        #     u_v_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
        #     u, v, p = u_v_p[:, 0:1], u_v_p[:, 1:2], u_v_p[:, 2:3]
            
        #     u_x, u_y = tape.gradient(u, [x, y])
        #     v_x, v_y = tape.gradient(v, [x, y])
        #     u_t = tape.gradient(u, t)
        #     v_t = tape.gradient(v, t)
        
        # u_xx, u_yy = tape.gradient([u_x, u_y], [x, y])
        # v_xx, v_yy = tape.gradient([v_x, v_y], [x, y])
        # p_x, p_y = tape.gradient(p, [x, y])
        
        # f_u = lambda_1 * (u_t + u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        # f_v = lambda_1 * (v_t + u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
        # cont = u_x + v_y  # Continuity equation
        # return u, v, p, f_u, f_v, cont
    
    def net_NS_data(self, x, y, t):
        # lambda_1 = self.lambda_1
        # lambda_2 = self.lambda_2
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        t = tf.cast(t, tf.float32)
        
        # with tf.GradientTape(persistent=True) as tape:
        #     tape.watch([x, y, t])
        u_v_p = self.neural_net(tf.concat([x, y, t], 1), self.nn_weights, self.nn_biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u,v,p

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
    # @tf.function(jit_compile=True)

    def get_cylinder_mask(self, x, y, center=(0, 0), diameter=2):
        """ Returns a mask for points inside the cylinder """
        r = diameter / 2.0
        mask = (x - center[0])**2 + (y - center[1])**2 <= r**2
        return mask

    # @tf.function(jit_compile=True, reduce_retracing=True)
    def loss_fn(self, x_batch, y_batch, t_batch, u_batch, v_batch, p_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        u_batch = tf.cast(u_batch, tf.float32)
        v_batch = tf.cast(v_batch, tf.float32)
        p_batch = tf.cast(p_batch, tf.float32)

        u_pred, v_pred, p_pred, f_u_pred, f_v_pred, cont = self.net_NS(x_batch, y_batch, t_batch)
        loss = tf.math.reduce_mean(tf.square(u_batch - u_pred)) + \
               tf.math.reduce_mean(tf.square(v_batch - v_pred)) + \
               tf.math.reduce_mean(tf.square(f_u_pred)) + \
               tf.math.reduce_mean(tf.square(f_v_pred)) + \
               tf.math.reduce_mean(tf.square(p_batch - p_pred)) + \
               tf.math.reduce_mean(tf.square(cont))

        # Compute loss for initial conditions
        u_i_pred, v_i_pred, p_i_pred = self.net_NS_data(self.x_i, self.y_i, self.t_i)
        loss_i = tf.math.reduce_mean(tf.square(self.u_i - u_i_pred)) + tf.math.reduce_mean(tf.square(self.v_i - v_i_pred)) + tf.math.reduce_mean(tf.square(self.p_i - p_i_pred))

        # Compute loss for boundary conditions
        u_b_pred, v_b_pred, p_b_pred = self.net_NS_data(self.x_b, self.y_b, self.t_b)
        loss_b = tf.math.reduce_mean(tf.square(self.u_b - u_b_pred)) + tf.math.reduce_mean(tf.square(self.v_b - v_b_pred)) + tf.math.reduce_mean(tf.square(self.p_b - p_b_pred))

        # mask = self.get_cylinder_mask(x_batch, y_batch)
        # u_inside_cylinder = tf.boolean_mask(u_pred, mask)
        # v_inside_cylinder = tf.boolean_mask(v_pred, mask)
        # cylinder_loss = tf.math.reduce_mean(tf.square(u_inside_cylinder)) + tf.math.reduce_mean(tf.square(v_inside_cylinder))

        total_loss = loss + loss_i + loss_b
        return total_loss
    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(self, inputs):
        x_batch, y_batch, t_batch, u_batch, v_batch, p_batch = inputs
        with tf.GradientTape() as tape1:
            loss = self.loss_fn(x_batch, y_batch, t_batch, u_batch, v_batch, p_batch)
        gradients = tape1.gradient(loss, [self.lambda_1, self.lambda_2] + self.nn_weights + self.nn_biases)
        self.optimizer.apply_gradients(zip(gradients, [self.lambda_1, self.lambda_2] + self.nn_weights + self.nn_biases))
        del tape1
        return loss


    def train(self, nIter, x_train, y_train, t_train, u_train, v_train, p_train, x_val, y_val, t_val, u_val, v_val, p_val, plot_interval=10000):
        start_time = time.time()
        # iterations = []
        self.loss_history = []
        # x_star = np.linspace(self.lb[0], self.ub[0], 200)[:, None]
        # y_star = np.linspace(self.lb[1], self.ub[1], 200)[:, None]
        # X_star_, Y_star_ = np.meshgrid(x_star, y_star)
        # x_star = X_star_.flatten()[:, None]
        # y_star = Y_star_.flatten()[:, None]

        # j = 0

        for it in range(nIter):
            loss = self.train_step((x_train, y_train, t_train, u_train, v_train, p_train))
            # loss_history.append(loss.numpy())
            if it % 400 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.numpy()
                lambda_2_value = self.lambda_2.numpy()
                u_val_pred, v_val_pred, p_val_pred, _, _, _ = self.net_NS(x_val, y_val, t_val)

                error_u_val = np.linalg.norm(u_val - u_val_pred, 2) / np.linalg.norm(u_val, 2)
                error_v_val = np.linalg.norm(v_val - v_val_pred, 2) / np.linalg.norm(v_val, 2)
                error_p_val = np.linalg.norm(p_val - p_val_pred, 2) / np.linalg.norm(p_val, 2)
                accuracy = 100 * (1 - (error_u_val + error_v_val + error_p_val) / 3)
                # strouhal_number = self.calculate_strouhal_number(v_val_pred, t_val, self.characteristic_length)
                self.data['iterations'].append(it)
                self.data['loss'].append(loss.numpy())
                self.data['accuracy'].append(accuracy)
                self.data['lambda_1'].append(lambda_1_value)
                self.data['lambda_2'].append(lambda_2_value)
                # self.data['strouhal_number'].append(strouhal_number)
                # iterations.append(it)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Val Error u: %.3e, v: %.3e, p: %.3e, Accuracy: %.2f%%, Time: %.2f' %
                    (it, loss, lambda_1_value, lambda_2_value, error_u_val, error_v_val, error_p_val, accuracy, elapsed))
                if lambda_1_value >= 0.99 and lambda_2_value >= 0.009 and lambda_1_value <= 1.1 and lambda_2_value <= 0.011 and accuracy > 95:
                    print(f"Stopping training as lambda_1: {lambda_1_value} and lambda_2: {lambda_2_value} reached the target values.")
                    break
                start_time = time.time()

            # Visualize predictions every 'plot_interval' iterations
            # if it % plot_interval == 0 and it > 0:
            #     j += 1
            #     self.predict_and_plot(x_star, y_star, [210], j)

        self.lbfgs_train(x_train, y_train, t_train, u_train, v_train, p_train)
        u_val_pred, v_val_pred, p_val_pred, _, _, _ = self.net_NS(x_val, y_val, t_val)

        error_u_val = np.linalg.norm(u_val - u_val_pred, 2) / np.linalg.norm(u_val, 2)
        error_v_val = np.linalg.norm(v_val - v_val_pred, 2) / np.linalg.norm(v_val, 2)
        error_p_val = np.linalg.norm(p_val - p_val_pred, 2) / np.linalg.norm(p_val, 2)
        accuracy = 100 * (1 - (error_u_val + error_v_val + error_p_val) / 3)
        print(f'Final accuracy of the model: {accuracy}')
        # loss_history.append(loss_2.numpy())
        # self.save_training_data("training_results")
        return self.loss_history

    def predict_and_plot(self, x_star, y_star, t_values, j, index_start=1):
        for i, t_val in enumerate(t_values):
            t_test = np.full((x_star.shape[0], 1), t_val)
            # t_test = (t_test - model.t_mean) / model.t_std  # Normalize time values
            u_pred, _, _ = model.predict(x_star, y_star, t_test)

            # Plot results
            plot_solution(np.concatenate([x_star, y_star], axis=1), u_pred, index=index_start + j)


    @tf.function(jit_compile=True, reduce_retracing=True)
    def lbfgs_train(self, x_train, y_train, t_train, u_train, v_train, p_train):
        def get_flat_params():
            params = [self.lambda_1, self.lambda_2] + self.nn_weights + self.nn_biases
            return tf.concat([tf.reshape(param, [-1]) for param in params], axis=0)

        def set_flat_params(flat_params):
            idx = 0
            for param in [self.lambda_1, self.lambda_2] + self.nn_weights + self.nn_biases:
                shape = tf.shape(param)
                size = tf.reduce_prod(shape)
                new_param = tf.reshape(flat_params[idx: idx + size], shape)
                param.assign(new_param)
                idx += size

        def loss_and_grads(flat_params):
            set_flat_params(flat_params)
            with tf.GradientTape() as tape2:
                loss = self.loss_fn(x_train, y_train, t_train, u_train, v_train, p_train)
            grads = tape2.gradient(loss, [self.lambda_1, self.lambda_2] + self.nn_weights + self.nn_biases)
            flat_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
            del tape2
            
            # Append the current L-BFGS loss to the loss history
            # self.loss_history.append(loss)

            
            return loss, flat_grads
        
        initial_params = get_flat_params()
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_grads,
            initial_position=initial_params,
            max_iterations=500
        )
        set_flat_params(results.position)

    def predict(self, x_star, y_star, t_star):
        u_star, v_star, p_star,_,_,_ = self.net_NS(x_star, y_star, t_star)
        return u_star.numpy(), v_star.numpy(), p_star.numpy()

    def validate_at_timestamps(self, x_val, y_val, t_val, u_val, v_val, time_stamps):
        for t in time_stamps:
            idx = np.where(t_val == t)[0]
            x_t = x_val[idx]
            y_t = y_val[idx]
            t_t = t_val[idx]
            u_t_actual = u_val[idx]
            v_t_actual = v_val[idx]
            print(u_t_actual.shape, v_t_actual.shape)
            u_t_pred, v_t_pred, _ = self.predict(x_t, y_t, t_t)

            mae_u = mean_absolute_error(u_t_actual, u_t_pred)
            mse_u = mean_squared_error(u_t_actual, u_t_pred)
            rmse_u = np.sqrt(mse_u)
            r2_u = r2_score(u_t_actual, u_t_pred)

            mae_v = mean_absolute_error(v_t_actual, v_t_pred)
            mse_v = mean_squared_error(v_t_actual, v_t_pred)
            rmse_v = np.sqrt(mse_v)
            r2_v = r2_score(v_t_actual, v_t_pred)

            print(f'Timestamp {t}:')
            print(f'  u: MAE={mae_u:.3e}, MSE={mse_u:.3e}, RMSE={rmse_u:.3e}, R2={r2_u:.3f}')
            print(f'  v: MAE={mae_v:.3e}, MSE={mse_v:.3e}, RMSE={rmse_v:.3e}, R2={r2_v:.3f}')

    def calculate_strouhal_number(self, u_star, t_star, characteristic_length):
        # Reshape the predicted velocity to have time on the second axis
        u_star = u_star.numpy()

        u_reshaped = u_star.reshape(-1, t_star.shape[0])

        # Calculate the mean velocity for normalization
        mean_velocity = np.mean(u_reshaped, axis=1, keepdims=True)
        u_normalized = u_reshaped - mean_velocity

        # Perform FFT to find the dominant frequency
        fft_result = np.fft.fft(u_normalized, axis=1)
        fft_frequencies = np.fft.fftfreq(t_star.shape[0], d=t_star[1] - t_star[0])
        dominant_frequency_idx = np.argmax(np.abs(fft_result), axis=1)
        dominant_frequency = np.abs(fft_frequencies[dominant_frequency_idx])

        # Calculate Strouhal number
        strouhal_number = dominant_frequency * characteristic_length / np.abs(mean_velocity).mean()
        return strouhal_number

def plot_solution(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()

def plot_loss_vs_iterations(loss_history):
    plt.figure(figsize=(10, 6))
    plt.ticklabel_format(axis='both', style='sci')
    plt.plot(range(len(loss_history)), loss_history, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations (Adam + L-BFGS)')
    plt.legend()
    plt.show()

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

def train_with_multiple_optimizers(x_train, y_train, t_train, u_train, v_train, p_train,
                                   x_val, y_val, t_val, u_val, v_val, p_val,
                                   layers, initial_weights, initial_biases,
                                   num_iterations=80000, characteristic_length=2.0):
    global model
    # Define the optimizers to test
    optimizers = {
        'Nadam': tf.keras.optimizers.Nadam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=22000,
            decay_rate=0.95,
            staircase=False
        )),
    }

    # Initialize dictionary to store loss and accuracy for each optimizer
    results = {optimizer_name: {'iterations': [], 'loss': [], 'accuracy': []} for optimizer_name in optimizers}

    for optimizer_name, optimizer in optimizers.items():
        print(f"\nTraining with {optimizer_name} optimizer...\n")
        
        # Initialize model
        model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers, 
                                  x_b, y_b, t_b, u_b, v_b, p_b, x_i, y_i, t_i, u_i, v_i, p_i, 
                                  characteristic_length, initial_weights=None, initial_biases=None)
        
        # Set the optimizer for this run
        model.optimizer = optimizer
        
        # Train the model and save the history
        loss_history = model.train(num_iterations, x_train, y_train, t_train, u_train, v_train, p_train,
                                   x_val, y_val, t_val, u_val, v_val, p_val)
        # print(f' For optimzer: {optimizer}, loss is {loss_history[-1]}')
        # Save loss and accuracy for this optimizer
        results[optimizer_name]['iterations'] = model.data['iterations']
        results[optimizer_name]['loss'] = model.data['loss']
        results[optimizer_name]['accuracy'] = model.data['accuracy']
        
        # plot_loss_vs_iterations(loss_history)
        # loss_history_tensor = tf.convert_to_tensor(loss_history)
        # loss_h = []
        # Print or plot the loss history
        # for i, loss_value in enumerate(loss_history):
        #     loss_h.append(loss_value.numpy())
        # results[optimizer_name]['loss_lbfgs'] = loss_h
        # print(f'Final loss value:{loss_h[-1]:.4e}')
    return results

# def plot_loss_accuracy(results):
#     plt.figure(figsize=(12, 6))
#     plt.ticklabel_format(axis='both', style='sci')
#     # # Plot Loss vs Iterations
#     # plt.subplot(1, 2, 1)
#     # for optimizer_name, data in results.items():
#     #     plt.plot(data['iterations'], data['loss'], label=f'{optimizer_name} Loss')
#     # plt.xlabel('Iterations')
#     # plt.ylabel('Loss')
#     # plt.title('Loss vs Iterations')
#     # plt.legend()
    
#     # Plot Accuracy vs Iterations
#     plt.subplot(1, 2, 1)
#     for optimizer_name, data in results.items():
#         plt.plot(data['iterations'], data['accuracy'], label=f'{optimizer_name} Accuracy')
#     plt.xlabel('Iterations')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Accuracy vs Iterations')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()
# def plot_loss_vs_iterations(loss_history):
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(loss_history)), loss_history, label='Adam + L-BFGS Loss')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.title('Loss vs Iterations (Adam + L-BFGS)')
#     plt.legend()
#     plt.show()

# Main execution

    # Assuming you've loaded your data already
characteristic_length = 2.0
initial_weights = vanilla_nn_weights
initial_biases = vanilla_nn_biases
print('1')
# Train with different optimizers
results = train_with_multiple_optimizers(x_train, y_train, t_train, u_train, v_train, p_train,
                                            x_val, y_val, t_val, u_val, v_val, p_val,
                                            lyers, initial_weights, initial_biases,
                                            num_iterations=150001)

# Plot the results including Adam + L-BFGS
# plot_loss_accuracy(results)
# print('2')
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('PINN_training_with_optimizers.csv', index=False)


# After training the vanilla neural network
# vanilla_nn.save('vanilla_nn_model.h5')


# Save the model architecture to a JSON file
# model_json = model.to_json()
# with open("pinn_model.json", "w") as json_file:
#     json_file.write(model_json)

# Save the model weights separately
# model.save_weights("pinn_model_weights.h5")
# import gc

# # Clear session and garbage collect
# tf.keras.backend.clear_session()
# gc.collect()
# # @tf.function(jit_compile=False)
# model.save('pinn_model.keras')
# # Now try saving the model
# # model.save('pinn_model', save_format='tf')
# # import numpy as np
# # import pandas as pd

# def predict_and_save(model, x_star, y_star, t_values, filename='pinn_predictions.csv'):
#     predictions = []

#     for t_val in t_values:
#         t_test = np.full((x_star.shape[0], 1), t_val)
#         u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_test)

#         # Store predictions along with corresponding t values
#         for i in range(x_star.shape[0]):
#             predictions.append([x_star[i, 0], y_star[i, 0], t_val, u_pred[i, 0], v_pred[i, 0], p_pred[i, 0]])

#     # Convert to DataFrame
#     df = pd.DataFrame(predictions, columns=['x', 'y', 't', 'u_pred', 'v_pred', 'p_pred'])

#     # Save to CSV
#     df.to_csv(filename, index=False)
#     print(f"Predictions saved to {filename}")

# # Define your spatial grid (x_star, y_star) for prediction
# x_star = np.linspace(model.lb[0], model.ub[0], 200)[:, None]
# y_star = np.linspace(model.lb[1], model.ub[1], 200)[:, None]
# X_star_, Y_star_ = np.meshgrid(x_star, y_star)
# x_star = X_star_.flatten()[:, None]
# y_star = Y_star_.flatten()[:, None]

# # Generate time values from t = 0 to t = 250
# t_values = np.arange(0, 251)  # Generates [0, 1, 2, ..., 250]

# # Predict and save the results
# predict_and_save(model, x_roi, y_roi, t_roi)
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_results(x_star, y_star, predictions, title, index):
    lb = np.min(x_star), np.min(y_star)
    ub = np.max(x_star), np.max(y_star)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)
    Z = griddata((x_star.flatten(), y_star.flatten()), predictions.flatten(), (X, Y), method='linear')

    plt.subplot(3, 1, index)
    plt.pcolor(X, Y, Z, cmap='jet')
    plt.colorbar()
    plt.title(title)

def predict_and_plot(model, x_star, y_star, t_values, u_star, v_star, p_star, t_roi, save_dir='plots'):
    
    for idx, t_val in enumerate(t_values):
        fig, axes = plt.subplots(3, figsize=(8,12))

        # t_test = np.full((x_star.shape[0], 1), t_val)
        idxp = np.where(t_val == t_roi)[0]
        u_t = u_star[idxp].reshape(-1,1)
        v_t = v_star[idxp].reshape(-1,1)
        p_t = p_star[idxp].reshape(-1,1)
        x_t = x_star[idxp].reshape(-1,1)
        print(x_t.shape)
        y_t = y_star[idxp].reshape(-1,1)
        t_t = t_roi[idxp].reshape(-1,1)
        u_pred, v_pred, p_pred = model.predict(x_t, y_t, t_t)
        

        # Calculate errors
        mae_u = mean_absolute_error(u_t, u_pred)
        mse_u = mean_squared_error(u_t, u_pred)
        # r2_u = r2_score(u_t, u_pred)

        mae_v = mean_absolute_error(v_t, v_pred)
        mse_v = mean_squared_error(v_t, v_pred)
        # r2_v = r2_score(v_t, v_pred)

        mae_p = mean_absolute_error(p_t, p_pred)
        mse_p = mean_squared_error(p_t, p_pred)
        # r2_p = r2_score(p_t, p_pred)

        # Plot u
        plot_results(x_t, y_t, u_pred, f'u_pred at t={t_val}\nMAE: {mae_u:.3e}, MSE: {mse_u:.3e}', 1)
        # Plot v
        plot_results(x_t, y_t, v_pred, f'v_pred at t={t_val}\nMAE: {mae_v:.3e}, MSE: {mse_v:.3e}', 2)
        # Plot p
        plot_results(x_t, y_t, p_pred, f'p_pred at t={t_val}\nMAE: {mae_p:.3e}, MSE: {mse_p:.3e}', 3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'predictions_at_t_{t_val}.png'))
        plt.close(fig)
    # 
    # plt.show()

# Example usage
# x_star = np.linspace(model.lb[0], model.ub[0], 200)[:, None]
# y_star = np.linspace(model.lb[1], model.ub[1], 200)[:, None]
# X_star_, Y_star_ = np.meshgrid(x_star, y_star)
# x_star = X_star_.flatten()[:, None]
# y_star = Y_star_.flatten()[:, None]

# Generate predictions and plots for t = 50, 100, 150, 200, 240
t_values = [50, 100, 158, 200, 240]
predict_and_plot(model, x_roi, y_roi, t_values, u_roi, v_roi, p_roi, t_roi)
print(f'The final lambda 1 value:{model.lambda_1}\n lambda 2 value:{model.lambda_2}')


from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

# Assuming your data arrays are loaded as follows:
# velocity.shape = (18840, 2, 400)
# time.shape = (1, 400)
# points.shape = (18840, 2, 400)

# Generate synthetic data for illustration
# np.random.seed(0)
velocity = U_star
tme = t_star
points = X_star

# Extract tme values (flatten the tme array)
tme = tme.flatten()

# Calculate mean or maximum velocity for each point
mean_velocities = np.mean(np.linalg.norm(velocity, axis=1), axis=1)
max_velocities = np.max(np.linalg.norm(velocity, axis=1), axis=1)

# Find the index of the point with the highest mean or maximum velocity
max_mean_velocity_index = np.argmax(mean_velocities)
max_max_velocity_index = np.argmax(max_velocities)

# Choose the criterion for selection (mean or max velocity)
selected_index = max_mean_velocity_index  # Change to max_mean_velocity_index if using mean velocity

# Extract the velocity tme series for the selected point
velocity_y_shedding = velocity[selected_index, 1, :]  # Focusing on y-component
print(mean_velocities.shape)
print(max_max_velocity_index)
print(velocity_y_shedding.shape)
# Plot the velocity tme series for the selected point
plt.figure(figsize=(10, 4))
plt.plot(tme, velocity_y_shedding)
plt.xlabel('time (s)')
plt.ylabel('Velocity in y-direction (m/s)')
plt.title(f'Velocity tme Series at Point Index {selected_index}')
plt.savefig(os.path.join('plots', f'St_data_v.png'))
plt.close()
# plt.show()

# Interpolate the velocity data to create a uniformly spaced tme series
uniform_tme = np.linspace(tme.min(), tme.max(), len(tme))
interpolator = interp1d(tme, velocity_y_shedding, kind='linear')
velocity_y_shedding_interpolated = interpolator(uniform_tme)

# Ensure the tme steps are uniform
dt = np.mean(np.diff(uniform_tme))
sampling_rate = 1 / dt

# Perform FFT
N = len(uniform_tme)
fft_result = fft(velocity_y_shedding_interpolated)
frequencies = fftfreq(N, dt)
print(tme.shape, velocity_y_shedding.shape)
# Take the absolute value of the FFT result to get the magnitude
magnitude = np.abs(fft_result)

# Find the peak frequency in the positive half of the frequencies (excluding the zero frequency)
positive_frequencies = frequencies[:N // 2]
positive_magnitude = magnitude[:N // 2]
nonzero_indices = np.where(positive_frequencies > 0.01)
peak_frequency_index = np.argmax(positive_magnitude[nonzero_indices])
peak_frequency = positive_frequencies[nonzero_indices][peak_frequency_index]

# Plotting the FFT result for verification
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, positive_magnitude)
plt.axvline(peak_frequency, color='r', linestyle='--', label=f'Peak Frequency: {peak_frequency:.2f} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Velocity time Series')
plt.legend()
plt.savefig(os.path.join('plots', f'St_data_fft.png'))
plt.close()

# Known values (Ensure these are correct for your simulation)
L = 2.0  # characteristic length (e.g., diameter of the cylinder) in meters
U = 1.0  # free-stream velocity in meters per second

# Calculate Strouhal number
St = peak_frequency * L / U

# Print the Strouhal number
print(f'Strouhal Number: {St}')


# Assuming model is already trained and loaded

# Extract the spatial location (x, y) at the selected index
x_selected = points[selected_index, 0, :]
y_selected = points[selected_index, 1, :]
print(x_selected.shape)
# Create a meshgrid or linspace for the times you want to predict
# Assuming time.shape is (1, N) where N is the number of time steps
time_pred = np.linspace(tme.min(), tme.max(), tme.shape[0])

# Prepare the input data for the model
x_input = np.tile(x_selected[:, None], (1, len(time_pred))).flatten()
y_input = np.tile(y_selected[:, None], (1, len(time_pred))).flatten()
t_input = np.tile(time_pred, (x_selected.shape[0], 1)).flatten()
print(x_input.shape, time_pred.reshape(-1, 1).shape)
# Use the model to predict the velocity at these points
u_pred, v_pred, p_pred = model.predict(x_selected.reshape(-1, 1), y_selected.reshape(-1, 1), time_pred.reshape(-1, 1))

# Extract the y-component of the velocity at the selected index
velocity_y_pred = v_pred
print(velocity_y_pred.shape,time_pred.shape)
# Interpolate the predicted velocity data to create a uniformly spaced time series
uniform_time_pred = np.linspace(time_pred.min(), time_pred.max(), len(time_pred))
interpolator_pred = interp1d(time_pred, velocity_y_pred.flatten(), kind='linear')
velocity_y_pred_interpolated = interpolator_pred(uniform_time_pred)

# Ensure the time steps are uniform
dt_pred = np.mean(np.diff(uniform_time_pred))
sampling_rate_pred = 1 / dt_pred
plt.figure(figsize=(10, 4))
plt.plot(time_pred, velocity_y_pred)
plt.xlabel('time (s)')
plt.ylabel('Velocity in y-direction (m/s)')
plt.title(f'Velocity tme Series at Point Index {selected_index}')
plt.savefig(os.path.join('plots', f'St_pred_v.png'))
plt.close()
# Perform FFT
N_pred = len(uniform_time_pred)
fft_result_pred = fft(velocity_y_pred_interpolated)
frequencies_pred = fftfreq(N_pred, dt_pred)

# Take the absolute value of the FFT result to get the magnitude
magnitude_pred = np.abs(fft_result_pred)

# Find the peak frequency in the positive half of the frequencies (excluding the zero frequency)
positive_frequencies_pred = frequencies_pred[:N_pred // 2]
positive_magnitude_pred = magnitude_pred[:N_pred // 2]
nonzero_indices_pred = np.where(positive_frequencies_pred > 0.01)
peak_frequency_index_pred = np.argmax(positive_magnitude_pred[nonzero_indices_pred])
peak_frequency_pred = positive_frequencies_pred[nonzero_indices_pred][peak_frequency_index_pred]

# Plotting the FFT result for verification
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies_pred, positive_magnitude_pred)
plt.axvline(peak_frequency_pred, color='r', linestyle='--', label=f'Peak Frequency: {peak_frequency_pred:.2f} Hz')
plt.ylim(0,2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Predicted Velocity Time Series')
plt.legend()
plt.savefig(os.path.join('plots', 'St_pred_fft'))
plt.close()
# Calculate Strouhal number for predicted data
St_pred = peak_frequency_pred * L / U

# Print the Strouhal number
print(f'Predicted Strouhal Number: {St_pred}')

relative_error = np.abs(St - St_pred) / St * 100
print(relative_error)
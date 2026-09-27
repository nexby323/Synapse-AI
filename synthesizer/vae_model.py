import tensorflow as tf 
from tensorflow.keras import layers, Model 
import numpy as np 
"""
    https://en.wikipedia.org/wiki/Variational_autoencoder
    https://medium.com/@jain.sm/autoencoders-and-variational-autoencoders-an-introduction-e37e82f1bbad
    Normal autoencoder is conssisting of 2 elements:  
    
    The encoder maps high dimensional data (raw data) into low dimension point in the latent space 
    the latent space is a n-dimensional space where "similar" data is found in clusters.
    also called f(x) (x is the data)
    
    The decoder maps the point from the latent space to the original data (at least trying the main importance is the training process)
    also called g(x)
     
    so it make sense that the cost function of a model from this type is something like L(x,g(f(x))). 
    
    VAE - Variational Auto Encoder 
    https://deeplearning.cs.cmu.edu/F24/document/slides/lec22.VAE.pdf
    in this case the encoder's output is the mean and the variance (log of variance but this is just for the purpose of dealing with small numbers)
    and from this output we get the Q_phi(Z,X) where z is the z_score of the normal probabilty 
    phi are the weights, from those normal distribution the model tries to decode the original input 
    and then the model evaluates the error and fix according to that the weights.
    when we inject to the cost function a component in the form of the KL divergence we get that the latent space is much more dense, in a way that any point that we take as an input 
    the decode of the point will output us a very similar data to the real one (from the statistical aspect). 
    from this reason VAE is fit for the purpose of generating similar but new data. 
     
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TabularVAE(Model):
    """
    Variational Autoencoder (VAE) for Tabular Data.
    Learns the underlying probability distribution of input features and generates 
    statistically similar, yet entirely synthetic, data points to ensure privacy.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4, **kwargs):
        super(TabularVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ==========================================
        # 1. ENCODER NETWORK (Inference Model q(z|x))
        # ==========================================
        # Maps the input data to a lower-dimensional hidden representation.
        self.encoder_inputs = layers.InputLayer(input_shape=(input_dim,))
        self.enc_dense1 = layers.Dense(16, activation='relu')
        self.enc_dense2 = layers.Dense(8, activation='relu')
        
        # ==========================================
        # 2. LATENT SPACE PROJECTION
        # ==========================================
        # Instead of outputting a single point, the encoder outputs parameters 
        # for a Gaussian distribution: Mean (μ) and Log-Variance (log(σ^2)).
        # We use log-variance for numerical stability (variance must be > 0, 
        # but neural networks output values from -∞ to +∞).
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        
        # ==========================================
        # 3. DECODER NETWORK (Generative Model p(x|z))
        # ==========================================
        # Reconstructs the original data format from the sampled latent vector 'z'.
        self.dec_dense1 = layers.Dense(8, activation='relu')
        self.dec_dense2 = layers.Dense(16, activation='relu')
        # Linear activation is used at the output assuming the input data is scaled/standardized.
        self.decoder_outputs = layers.Dense(input_dim, activation='linear') 

    def sample(self, z_mean, z_log_var):
        """
        Reparameterization Trick: z = μ + ε * σ
        Decouples the random sampling process from the network's weights, 
        allowing gradients to flow backward through the deterministic nodes.
        """
        # Extract batch size and latent dimension dynamically
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample standard Gaussian noise: ε ~ N(0, I)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Calculate standard deviation: σ = exp(0.5 * log(σ^2))
        # and apply the reparameterization formula.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        """
        Defines the forward pass of the network.
        """
        # Pass input through the encoder layers
        x = self.encoder_inputs(inputs)
        x = self.enc_dense1(x)
        x = self.enc_dense2(x)
        
        # Obtain the distribution parameters for this specific input
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        # During training, sample 'z' stochastically. 
        # During inference (if predicting), use the deterministic mean.
        if training:
            z = self.sample(z_mean, z_log_var)
        else:
            z = z_mean
            
        # Reconstruct the data from the latent representation
        x_decoded = self.dec_dense1(z)
        x_decoded = self.dec_dense2(x_decoded)
        reconstructed_output = self.decoder_outputs(x_decoded)
        
        return reconstructed_output, z_mean, z_log_var

    def train_step(self, data):
        """
        Custom training logic implementing the ELBO (Evidence Lower Bound) loss function.
        """
        with tf.GradientTape() as tape:
            # 1. Forward pass
            reconstruction, z_mean, z_log_var = self(data, training=True)
            
            # 2. Reconstruction Loss
            # Measures how well the decoder recreated the input (Mean Squared Error).
            # Multiplied by input_dim to sum the error across all features.
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(data, reconstruction)
            ) * self.input_dim
            
            # 3. KL Divergence Loss
            # Penalizes distributions that deviate from the standard normal N(0, I).
            # Formula: -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            # 4. Total Loss
            total_loss = reconstruction_loss + kl_loss
            
        # Compute gradients and update network weights via backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Return metrics for monitoring
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    def generate_synthetic(self, num_samples: int) -> np.ndarray:
        """
        Bypasses the encoder entirely to generate novel data.
        Samples random coordinates from the latent space and decodes them.
        """
        # Sample pure noise from the standard normal prior distribution
        random_latent_vectors = tf.random.normal(shape=(num_samples, self.latent_dim))
        
        # Push the random coordinates through the trained decoder
        x_decoded = self.dec_dense1(random_latent_vectors)
        x_decoded = self.dec_dense2(x_decoded)
        generated_data = self.decoder_outputs(x_decoded)
        
        return generated_data.numpy()
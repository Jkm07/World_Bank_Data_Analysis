import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans


class VAE(tf.keras.Model):

  def __init__(self, no_time_series = 128):
    super(VAE, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.001)
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(no_time_series, 31, 1)),
            tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 3), strides=(1, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 3), strides=(1, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 3), strides=(1, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(no_time_series * 2, activation = 'relu'),
            tf.keras.layers.Dense(no_time_series * 2),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape = (no_time_series, )),
            tf.keras.layers.Dense(no_time_series * 2, activation = 'relu'),
            tf.keras.layers.Dense(no_time_series * 8, activation = 'relu'),
            tf.keras.layers.Reshape(target_shape=(no_time_series, 1, 8)),
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu'),
            tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 1))),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=False)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  result = tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)
  return result


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  error = tf.keras.losses.MSE(y_true = x, y_pred = x_logit)
  logpx_z = -tf.reduce_sum(error, axis=[1, 2])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train(model, X, epochs):
  for epoch in range(1, epochs + 1):
    for train_x in X:
        train_x = tf.expand_dims(train_x, axis=0)
        train_step(model, train_x)
    loss = tf.keras.metrics.Mean()
    if epoch % 25 == 1:
        for test_x in X:
            test_x = tf.expand_dims(test_x, axis=0)
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}'
                .format(epoch, elbo))
  
def prepare_data(df):
  X = []
  Y = []
  for country_name in df['Country Name'].unique():
    Y.append(country_name)
    X.append(df[df['Country Name'] == country_name].loc[:,"1991":"2021"].to_numpy())
  X = np.array(X)
  X = np.expand_dims(X, axis = -1)
  return (X, Y)

def get_enoder_cluster(model, X, Y):
    X_encoded, _ = model.encode(X)
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_encoded)
    result = pd.Series(data=kmeans.labels_, index=Y)
    result.index.name = 'Country Name'
    return result
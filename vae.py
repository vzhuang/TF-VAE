import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

class VAE(object):
    """
    Variational autoencoder implementation using MLPs
    """
    def __init__(self, layer_sizes, input_dim=784, latent_dim=10, lr=0.005, batch_size=100):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # initialize autoencoder network weights
        self.init_weights(*layer_sizes)

        self.x = tf.placeholder(tf.float32, [None, input_dim])
                
        # encoder
        enc_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['enc1']),
                                  self.biases['enc1']))
        enc_2 = tf.nn.relu(tf.add(tf.matmul(enc_1, self.weights['enc2']),
                                  self.biases['enc2']))
        # encoder output - factorized Gaussian parameters
        mu = tf.add(tf.matmul(enc_2, self.weights['mu']) , self.biases['mu'])
        # ensure positive variance by training encoder on log of variance
        log_var = tf.add(tf.matmul(enc_2, self.weights['log_var']),
                         self.biases['log_var'])
        
        # compute samples using reparameterization trick
        z_samples = tf.random_normal(tf.shape(mu))
        z = tf.add(mu, tf.mul(tf.exp(0.5 * log_var), z_samples))

        # decoder
        dec_1 = tf.nn.relu(tf.add(tf.matmul(z, self.weights['dec1']),
                                  self.biases['dec1']))
        dec_2 = tf.nn.relu(tf.add(tf.matmul(dec_1, self.weights['dec2']),
                                  self.biases['dec2']))

        # decoder output
        self.output = tf.add(tf.matmul(dec_2, self.weights['out']), self.biases['out'])

        self.sigmoid_output = tf.sigmoid(self.output)
        
        # define loss 
        latent_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) -
                                           tf.exp(log_var), 1)
        # we use Bernoulli output in the decoder
        reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            self.output, self.x), 1)

        self.loss = tf.reduce_mean(latent_loss + reconstruction_loss)
        loss_summ = tf.scalar_summary("lower bound", self.loss)

        # create train op using vanilla SGD optimizer
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # add op for merging summary
        self.summary_op = tf.merge_all_summaries()

        # add Saver ops
        saver = tf.train.Saver()


    def init_weights(self, n_enc1, n_enc2, n_dec1, n_dec2):
        w_enc1 = weight_variable([self.input_dim, n_enc1])
        b_enc1 = bias_variable([n_enc1])
        w_enc2 = weight_variable([n_enc1, n_enc2])
        b_enc2 = bias_variable([n_enc2])
        w_mu = weight_variable([n_enc2, self.latent_dim])
        b_mu = bias_variable([self.latent_dim])
        w_log_var = weight_variable([n_enc2, self.latent_dim])
        b_log_var = bias_variable([self.latent_dim])
        w_dec1 = weight_variable([self.latent_dim, n_dec1])
        b_dec1 = bias_variable([n_dec1])
        w_dec2 = weight_variable([n_dec1, n_dec2])
        b_dec2 = bias_variable([n_dec2])
        w_dec_out = weight_variable([n_dec2, self.input_dim])
        b_dec_out = bias_variable([self.input_dim])
        self.weights = {'enc1': w_enc1,
                        'enc2': w_enc2,
                        'mu': w_mu,
                        'log_var': w_log_var,
                        'dec1': w_dec1,
                        'dec2': w_dec2,
                        'out': w_dec_out
        }
        self.biases = {'enc1': b_enc1,
                       'enc2': b_enc2,
                       'mu': b_mu,
                       'log_var': b_log_var,
                       'dec1': b_dec1,
                       'dec2': b_dec2,
                       'out': b_dec_out
        }

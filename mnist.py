import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from vae import VAE

batch_size = 100

# load MNIST
mnist = input_data.read_data_sets('MNIST')

vae = VAE([256, 256, 256, 256], lr=0.005, batch_size=batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.train.SummaryWriter('logs', graph=sess.graph)

epoch = 0    
step = 0
train_error = []
test_error = []
epoch_loss = 0
while mnist.train.epochs_completed <= 25:        
    step += 1
    if mnist.train.epochs_completed > epoch:
        epoch = mnist.train.epochs_completed
        print(epoch, curr_loss)            
        train_error.append(epoch_loss / step)
        epoch_loss = 0
        step = 0
        # compute test_error
        error = 0
        s = 0
        while mnist.test.epochs_completed < epoch:
            s += 1
            tb = mnist.test.next_batch(batch_size)
            for arr in tb[0]:
                arr[arr > 0] = 1;
            batch_loss = sess.run([vae.loss], feed_dict={vae.x: tb[0]})
            error += batch_loss[0]
        test_error.append(error / s)
    batch = mnist.train.next_batch(batch_size)
    # binarize
    for arr in batch[0]:
        arr[arr > 0] = 1.
    feed_dict = {vae.x: batch[0]}
    _, curr_loss, summary_str = sess.run([vae.train_step, vae.loss, vae.summary_op], feed_dict=feed_dict)
    epoch_loss += curr_loss
    summary_writer.add_summary(summary_str, step)

# visualize reconstructions
# code from https://jmetzen.github.io/2015-11-27/vae.html
x_sample, y = mnist.test.next_batch(1000)
for arr in x_sample:
    arr[arr > 0] = 1.
test_inputs = np.zeros([10, 784])
for i in range(10):
    test_inputs[i] = np.array(x_sample[np.where(y==i)[0][0]])
x_reconstruct = sess.run(vae.output, feed_dict={vae.x: test_inputs})
plt.figure(figsize=(4, 12))
for i in range(10):

    plt.subplot(10, 2, 2*i + 1)
    plt.imshow(test_inputs[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(10, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.savefig('vaereconstruction.png')

plt.style.use('ggplot')

print(test_error)
print(train_error
)
plt.figure(figsize=(6, 4))
plt.clf()
plt.xlabel('epoch')
plt.ylabel('test error')
plt.plot(range(25), test_error)
plt.savefig('vaetesterror.png')
plt.clf()
plt.xlabel('epoch')
plt.ylabel('training error')
plt.plot(range(25), train_error)
plt.savefig('vaetrainerror.png')

#Import necessary libraries
import tensorflow as tf
import numpy as np

#Set up GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

#Load model
saver = tf.train.import_meta_graph('model.ckpt.meta')

#Set up placeholders for input and output data
X = tf.get_default_graph().get_tensor_by_name("X:0")
Y = tf.get_default_graph().get_tensor_by_name("Y:0")

#Set up output layer
output = tf.get_default_graph().get_tensor_by_name("output/BiasAdd:0")

#Set up loss function and optimization
loss = tf.losses.mean_squared_error(Y, output) + tf.losses.get_regularization_loss() # include regularization loss


# Set up optimizer
optimizer = tf.train.AdamWOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Set up saver to save model
saver = tf.train.Saver()

# Load training data
train_data = np.load("train_data.npy")

# Set up session and run model
with tf.Session(config=config) as sess:
    # Load saved model
    saver.restore(sess, "model.ckpt")

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle training data
        np.random.shuffle(train_data)
        # Split into batches
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            batch_x = train_data[i * batch_size:(i + 1) * batch_size, :-1]
            batch_y = train_data[i * batch_size:(i + 1) * batch_size, -1]
            batch_y = batch_y.reshape(-1, 1)
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        # Calculate and print loss after each epoch
        train_loss = sess.run(loss, feed_dict={X: train_data[:, :-1], Y: train_data[:, -1].reshape(-1, 1)})
        print("Epoch {}: train loss = {}".format(epoch + 1, train_loss))
    # Save model after training is complete
    saver.save(sess, "model.ckpt")
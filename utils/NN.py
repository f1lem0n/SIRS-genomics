"""Neural Networks"""

from time import perf_counter
import os

from tensorflow.keras import layers
import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Deep Convoluted Generative Adversarial Network"""

    def __init__(self, checkpoint_dir=None, noise_dim=100,
                 save_checkpoints=False, batch_size=1,
                 loss_fun=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 optimizer=tf.keras.optimizers.legacy.Adam(1e-4)):
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.loss_function = loss_fun
        self.gen_optimizer = optimizer
        self.dis_optimizer = optimizer

    def make_generator(self):
        """Generator Network"""
        model = tf.keras.Sequential()
        model.add(layers.Dense(
            tf.reduce_prod(self.data_shape_)*256,
            use_bias=False,
            input_shape=(100,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((self.data_shape_ + (256,))))
        assert model.output_shape == (None,) + self.data_shape_ + (256,)

        model.add(layers.Conv2DTranspose(
            128, (5, 5), padding='same', use_bias=False
        ))
        assert model.output_shape == (None,) + self.data_shape_ + (128,)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(
            64, (5, 5), padding='same', use_bias=False
        ))
        assert model.output_shape == (None,) + self.data_shape_ + (64,)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(
            1, (5, 5), padding='same', use_bias=False, activation='tanh'
        ))
        assert model.output_shape == (None,) + self.data_shape_ + (1,)

        return model

    def make_discriminator(self):
        """Discriminator Network"""
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            64, (5, 5), strides=(2, 2),
            padding='same',
            input_shape=self.data_shape_ + (1,)
        ))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def _generator_loss(self, fake_output):
        return self.loss_function(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_function(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_function(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def _train_step(self, batch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_(noise, training=True)

            real_output = self.discriminator_(batch, training=True)
            fake_output = self.discriminator_(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator_.trainable_variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator_.trainable_variables
            )

            self.gen_optimizer.apply_gradients(zip(
                gradients_of_generator, self.generator_.trainable_variables
            ))
            self.dis_optimizer.apply_gradients(zip(
                gradients_of_discriminator,
                self.discriminator_.trainable_variables
            ))
            return gen_loss + disc_loss

    def fit(self, X, epochs, print_status=True):
        """Train DCGAN"""
        self.data_shape_ = (self.batch_size, X.shape[1])
        self.generator_ = self.make_generator()
        self.discriminator_ = self.make_discriminator()
        self.avg_losses_ = []

        if self.save_checkpoints:
            assert self.checkpoint_dir is not None
            checkpoint_prefix = os.path.join(
                self.checkpoint_dir, "checkpoint_"
            )
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.optimizer,
                discriminator_optimizer=self.optimizer,
                generator=self.generator_,
                discriminator=self.discriminator_
            )

        batches = []
        for idx, _ in enumerate(X):
            if idx % self.batch_size == 0:
                batch = X[idx:idx+self.batch_size]
                batches.append(batch)

        if print_status:
            print(f"{'='*16} TRAINING STATUS {'='*16}")
        for epoch in range(epochs):
            start = perf_counter()
            losses = []
            for batch in batches:
                losses.append(self._train_step(batch.reshape(
                    (self.batch_size,) * 2 + (X.shape[1], self.batch_size)
                )))
            if self.save_checkpoints:
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)
            loss = np.average(losses)
            self.avg_losses_
            finish = perf_counter()
            if print_status:
                print(f"[epoch {epoch + 1} | time: {finish - start:.2f} s | average loss: {loss:.5f}]")
        if print_status:
            print("="*33)

    def generate(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            sample = self.generator_(noise, training=False)[0, :, :, 0][0]
            samples.append(sample)
        return np.array(samples)

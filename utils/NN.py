"""Neural Networks"""

from time import perf_counter
import os

from tensorflow.keras import layers
import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Generative Adversarial Network"""

    def __init__(self, checkpoint_dir=None, noise_dim=100,
                 save_checkpoints=False, batch_size=1,
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 optimizer=tf.keras.optimizers.legacy.Adam(1e-4)):
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.loss = loss
        self.gen_optimizer = optimizer
        self.dis_optimizer = optimizer

    def make_generator(self):
        """Generator Network"""
        model = tf.keras.Sequential()
        model.add(layers.Dense(
            tf.reduce_prod(self._data_shape)*256,
            use_bias=False,
            input_shape=(100,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((self._data_shape + (256,))))
        assert model.output_shape == (None,) + self._data_shape + (256,)

        model.add(layers.Conv2DTranspose(128, (5, 5), padding='same', use_bias=False))
        assert model.output_shape == (None,) + self._data_shape + (128,)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, (5, 5), padding='same', use_bias=False))
        assert model.output_shape == (None,) + self._data_shape + (64,)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(
            1, (5, 5), padding='same', use_bias=False, activation='tanh'
        ))
        assert model.output_shape == (None,) + self._data_shape + (1,)

        return model

    def make_discriminator(self):
        """Discriminator Network"""
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            64, (5, 5), strides=(2, 2),
            padding='same',
            input_shape=self._data_shape + (1,)
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
        return self.loss(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def _train_step(self, batch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator(noise, training=True)

            real_output = self._discriminator(batch, training=True)
            fake_output = self._discriminator(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self._generator.trainable_variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self._discriminator.trainable_variables
            )

            self.gen_optimizer.apply_gradients(zip(
                gradients_of_generator, self._generator.trainable_variables
            ))
            self.dis_optimizer.apply_gradients(zip(
                gradients_of_discriminator, self._discriminator.trainable_variables
            ))

    def fit(self, X, epochs):
        """Train DCGAN"""
        self._data_shape = (self.batch_size, X.shape[1])
        self._generator = self.make_generator()
        self._discriminator = self.make_discriminator()

        if self.save_checkpoints:
            assert self.checkpoint_dir is not None
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint_")
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.optimizer,
                discriminator_optimizer=self.optimizer,
                generator=self._generator,
                discriminator=self._discriminator
            )

        batches = []
        for idx, _ in enumerate(X):
            if idx % self.batch_size == 0:
                batch = X[idx:idx+self.batch_size]
                batches.append(batch)

        for epoch in range(epochs):
            start = perf_counter()
            for batch in batches:
                self._train_step(batch.reshape(
                    (self.batch_size,) * 2 + (X.shape[1], self.batch_size)
                ))
            if self.save_checkpoints:
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)
            finish = perf_counter()
            print(f"[epoch {epoch + 1}: {finish - start} s]")

    def generate(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            sample = self._generator(noise, training=False)[0, :, :, 0][0]
            samples.append(sample)
        return np.array(samples)

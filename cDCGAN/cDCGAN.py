import os
import shutil
from typing import TypedDict

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import regularizers, Sequential
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, LayerNormalization, LeakyReLU, Reshape)
from ray import train
from sklearn.preprocessing import MinMaxScaler

# Have to create this as a packaged pyproject to use "riskray.ai.dicom..."
from ..dicom.dataframes import TrainingDataFrame
from ..dicom.dataset_tools import center_crop_df, downscale, META_DATA_KEYS, pad_df
from ..settings import KERAS_SEQUENTIAL, SEED, STORE_PATH


def load_keras_model(path):
    from keras.models import load_model
    return load_model(path)


class DiscriminatorConfig(TypedDict):
    filters_0: int
    kernel_0: int
    filters_1: int
    kernel_1: int


class GeneratorConfig(TypedDict):
    # Total latent dimension is latent_edge * latent_edge * (latent_channels + num_classes)
    latent_edge: int
    latent_channels: int
    num_classes: int
    # For the Conv2DTransposes
    kernel: int  # Might as well just keep it fixed
    out_0: int
    out_1: int
    out_2: int
    final_kernel: int  # A final reinterpretation of the full scale image-sized layer


# TODO -- f"MW Make this in Mixed Data Mode

def risk_ray_gan_discriminator(config: DiscriminatorConfig, trainer: "GANTrainer") -> Sequential:
    """
    Return our Keras cDCGAN Discriminator model
    """
    # Create discriminator
    return Sequential(
        [
            Conv2D(
                config['filters_0'], config['kernel_0'], strides=2, padding='same',
                input_shape=(trainer.input_shape[0], trainer.input_shape[1], trainer.n_channel),
                kernel_regularizer=regularizers.l2(0.01),
            ),
            LeakyReLU(alpha=0.2),
            Conv2D(
                config['filters_1'], config['kernel_1'], strides=2, padding='same',
                kernel_regularizer=regularizers.l2(0.01),
            ),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )


def risk_ray_gan_generator(config: GeneratorConfig) -> Sequential:
    """
    Return our Keras cDCGAN Generator model
    Note: It seems this source
    https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8
    does more processing on the label input [embedding -> dense, etc.]

    With padding='same', zeros are added to the outside s.t. we can go to the corners/edges.
    So, after the ConvTranspose, our square goes from [x, y] -> [x*stride, y*stride]
    """
    # Create generator
    # TODO -- f"MW Controlled Update
    # generator_input = int(config['latent_channels'] + config['num_classes'])
    generator_input = config['latent_channels']
    latent_edge = config['latent_edge']
    return Sequential(
        [
            Dense(
                latent_edge * latent_edge * generator_input,
                input_shape=(generator_input,)
            ),
            LayerNormalization(),
            LeakyReLU(alpha=0.2),
            Reshape((latent_edge, latent_edge, generator_input)),
            Conv2DTranspose(config['out_0'], config['kernel'], strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(config['out_1'], config['kernel'], strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(alpha=0.2),
            # Conv2DTranspose(config['out_2'], config['kernel'], strides=2, padding='same', use_bias=False),
            # LayerNormalization(),
            # LeakyReLU(alpha=0.2),
            Conv2D(1, config['final_kernel'], padding='same', activation='tanh'),
        ],
        name="generator",
    )


class cDCGAN(keras.Model):
    """
    Controlled Deep-Convolutional Generative Neural Net
    """

    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.combined_loss = keras.metrics.Mean(name="combined_loss")
        self.gen_sample_accuracy = keras.metrics.Accuracy(name="gen_sample_accuracy")
        self.gen_sample_mean = keras.metrics.Mean(name="gen_sample_mean")
        self.real_sample_accuracy = keras.metrics.Accuracy(name="real_sample_accuracy")
        self.real_sample_mean = keras.metrics.Mean(name="real_sample_mean")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def alt_train_step(self, real_images):
        """
        Training based off the TF website for DCGAN. In one experiment, I found it performed worse than
        with the other one. In fact it had trouble learning at all...
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            real_labels = tf.zeros((batch_size, 1))
            # Add random noise to the labels - important trick!
            real_labels = 0.05 * tf.random.uniform(tf.shape(real_labels))

            fake_output = self.discriminator(generated_images, training=True)
            fake_labels = tf.ones((batch_size, 1))
            fake_labels = 0.05 * tf.random.uniform(tf.shape(fake_labels))

            # Combine them with real images
            combined_prediction = tf.concat([fake_output, real_output], axis=0)
            combined_labels = tf.concat([fake_labels, real_labels], axis=0)

            gen_loss = self.loss_fn(fake_output, fake_labels)
            disc_loss = self.loss_fn(combined_prediction, combined_labels)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(disc_loss)
        self.g_loss_metric.update_state(gen_loss)
        # Report the metric to raytune
        metrics = {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
        metrics["combined_loss"] = metrics['d_loss'] + metrics['g_loss']
        return metrics

    @tf.function
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels = 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        # Report the metric to raytune
        metrics = {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
        metrics["combined_loss"] = metrics['d_loss'] + metrics['g_loss']
        return metrics


class GANMonitor(keras.callbacks.Callback):
    """
    Callback to save GAN generated images
    """

    def __init__(self, run_name, real_image_batch, num_img=3, latent_dim=128):
        """
        run_name actually has format "experiment/run" already
        """
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.run_name = run_name
        self.real_image_batch = real_image_batch

    def _eval_accuracy(self):
        """
        Custom method
        Evaluate the generator's performance on the discrim.
        We generate 20 fake images from latent vectors
        We also have a single training batch of real images to also eval discrim on.
        """
        # Sample random points in the latent space
        real_images = next(self.real_image_batch)
        real_batch = tf.shape(real_images)[0]
        fake_batch = 20
        random_latent_vectors = tf.random.normal(shape=(fake_batch, self.latent_dim))

        # Decode them to fake images
        generated_images = self.model.generator(random_latent_vectors, training=False)
        gen_predictions = self.model.discriminator(generated_images, training=False)
        real_predictions = self.model.discriminator(real_images, training=False)

        gen_sample_mean = self.model.gen_sample_mean(gen_predictions)
        real_sample_mean = self.model.real_sample_mean(real_predictions)

        gen_labels = tf.ones((fake_batch, 1))
        real_labels = tf.zeros((real_batch, 1))
        gen_accuracy = self.model.gen_sample_accuracy(tf.round(gen_predictions), gen_labels)
        real_accuracy = self.model.real_sample_accuracy(tf.round(real_predictions), real_labels)

        # Return the evaluated metrics
        return {
            "gen_sample_accuracy": gen_accuracy.numpy(),
            "real_sample_accuracy": real_accuracy.numpy(),
            "gen_sample_mean": gen_sample_mean.numpy(),
            "real_sample_mean": real_sample_mean.numpy()
        }

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        # generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            path = f"{STORE_PATH}/generated_images/{self.run_name}/gen_{epoch:03d}_{i}.png"
            img.save(path)

        # Generate and report metrics to RayTune
        g_loss, d_loss = self.model.d_loss_metric.result().numpy(), self.model.g_loss_metric.result().numpy()
        metrics = {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "combined_loss": d_loss + g_loss,
        }
        eval_metrics = self._eval_accuracy()
        metrics.update(**eval_metrics)

        # Also checkpoint the model
        checkpoint_dir = f"{STORE_PATH}/experiments/{self.run_name}/checkpoints"

        ray_ckpt = train.Checkpoint(checkpoint_dir)
        ray_ckpt.to_directory(checkpoint_dir)
        # print(f"MW RAY DEBUG -- {ray_ckpt.as_directory()}")

        train.report(metrics, checkpoint=ray_ckpt)

        # Save the keras model every 10 epochs
        if epoch % 10 == 0:
            model_dir = f"{STORE_PATH}/experiments/{self.run_name}/models/checkpoint_{epoch:06d}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # Save our keras models
            self.model.generator.save(f"{model_dir}/generator.keras", save_format='keras')
            self.model.discriminator.save(f"{model_dir}/discriminator.keras", save_format='keras')

            # Also save this checkpoint in a TF format
            # User "restore()" later
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.model.g_optimizer,
                discriminator_optimizer=self.model.d_optimizer,
                generator=self.model.generator,
                discriminator=self.model.discriminator
            )
            checkpoint.save(file_prefix=model_dir)


class GANTrainer:
    """
    Trainer for cDCGAN
    Takes in a TrainingDataFrame instance on init.

    Note that we're going to want KERAS_SEQUENTIAL=True before we start adding MixedData style.

    Example:
        df = DefaultDataFrame()
        gan = GANTrainer(df)
    """
    # Identifier the training session
    name = None

    # Identify this run. Default to self.name
    run = None

    # Learning datasets
    train_input, train_targets, test_input, test_targets = None, None, None, None
    train_attrs, test_attrs = None, None

    # TrainingDataFrame object for access the pandas store
    dataframe = None

    # The downscale_factor controls image resolution reduction.
    # A value of 2 is every other pixel, 4 is every 4th pixel and so on. 1 is neutral
    downscale_factor = 1

    # The image_reshape dimensions dictate the initial crop + pad normalization of the raw dataframe PixelArray objects
    # Note: This is not the final shape going into the neural net since downscaling will be
    # applied to images before that
    # image_reshape = (800, 700)
    image_reshape = (1000, 800)

    # Whereas image_reshape dictates how to crop / pad the preprocessed images,
    # input_shape captures the final shape after downsampling so that we can set the correct number of
    # input weights on our NN model
    input_shape = None
    # Color channels of an image. For RiskRay this is 1
    n_channel = 1

    # Meta Attributes
    meta_attrs, n_meta_attrs = None, None

    #
    # Some hyper-parameters
    #
    params = {
        "test_train_ratio": 0.2,
        "learning_rate": 0.01,
        "epochs": 10,
        "validation_split": 0.15
    }

    def __init__(self, dataframe: TrainingDataFrame, run=None, overwrite=True, **kwargs):
        """
        :param run:         Specify a learning session name. Defaults to the dataframe.name
        :param overwrite:   If True, overwrite certain directories, otherwise things may build up fast
        """
        assert issubclass(type(dataframe), TrainingDataFrame), f"Dataframe {dataframe} must subclass TrainingDataFrame"
        # Make sure dataframe is an instantiated object
        assert isinstance(dataframe, TrainingDataFrame), "Please provide an instantiated object for 'dataframe'"

        self.dataframe = dataframe
        self.name = dataframe.name
        self.run = run or self.name
        self.img_path = STORE_PATH + f"/images/{self.name}/{self.run}"
        self.store_path = STORE_PATH + f"/training/{self.name}/{self.run}"
        processed_data_path_root = self.store_path + "/processed_data"
        self.processed_data_paths = {
            "train_input": processed_data_path_root + "/train_input.npy",
            "train_attrs": processed_data_path_root + "/train_input_attrs.npy",
            "train_targets": processed_data_path_root + "/train_targets.npy",
            "test_input": processed_data_path_root + "/test_input.npy",
            "test_attrs": processed_data_path_root + "/test_input_attrs.npy",
            "test_targets": processed_data_path_root + "/test_targets.npy"
        }

        # Create store subdirectories if it doesn't exist
        if overwrite or not os.path.exists(self.store_path):
            print(f"Creating Training Store path {self.store_path}")
            if os.path.exists(self.store_path):
                shutil.rmtree(self.store_path)
            os.makedirs(self.store_path)

        if not os.path.exists(processed_data_path_root):
            # Make it
            os.makedirs(processed_data_path_root)

        # Make path for our generated images
        gen_path = f"{STORE_PATH}/generated_images/{self.run}"
        if not os.path.exists(gen_path):
            # Make it
            os.makedirs(gen_path)

        # Image parameters
        self.downscale_factor = kwargs.get('downscale_factor', self.downscale_factor)
        self.image_reshape = kwargs.get('image_reshape', self.image_reshape)
        self.n_channel = kwargs.get('n_channel', self.n_channel)
        self.meta_attrs = [m[0] for m in META_DATA_KEYS if m[1]]
        self.n_meta_attrs = len(self.meta_attrs)

        # So that we don't preprocess accidentally again
        self._preprocess_complete = False

        # So that we don't keep re-loading the dataframe
        if self.dataframe.dataframe is None:
            self._loaded_df = False
            self.load_dataframe()

        self._loaded_df = True

    def load_dataframe(self, reload=False):
        """
        Load the dataframe. Use reload boolean to force reload
        """
        if self._loaded_df and not reload:
            return

        # Pickle sets the self.dataframe.dataframe attribute for use
        self.dataframe.get_pickle()
        self._loaded_df = True

    def populate_learning_data(self, save=True, cleanup=True):
        """
        Populate the four learning arrays. Also does normalization to place pixels on range [0,1] inclusive

        With save=True, save the processed images to disk
        """
        print("Populating learning dataset")
        # Check if dataframe pickle loaded or not
        if self.dataframe.dataframe is None:
            self.load_dataframe()

        # Perform cropping and padding
        processed: pd.DataFrame = center_crop_df(
            self.dataframe.dataframe, 'PixelArray', self.image_reshape[0], self.image_reshape[1]
        )
        processed = pad_df(processed, 'PixelArray', self.image_reshape[0], self.image_reshape[1])

        # Perform downscaling
        processed: pd.DataFrame = downscale(processed, 'PixelArray', self.downscale_factor)

        train_input, train_targets = processed[['PixelArray']], processed['Target']
        # train_input, test_input, train_targets, test_targets = train_test_split(
        #     processed[['PixelArray', 'MetaAttrs']],
        #     processed['Target'],
        #     test_size=self.params['test_train_ratio'],
        #     random_state=SEED
        # )

        if cleanup:
            # Clear the dataframe and 'processed' object to release some memory
            del processed
            self.clear_dataframe()

        # Split dataset
        # Let's have our test set be 50% one-hots
        # counts = processed['Target'].value_counts()
        # n_hot = counts[1]
        # n_not = counts[0]
        # n_total = n_hot + n_not
        # n_test = max(10, int(n_total * self.params['test_train_ratio']))
        # n_train = n_total - n_test
        #
        # # Slice our dataframe
        # at_risk = processed.index[processed['Target'] == 1].tolist()
        # control = processed.index[processed['Target'] == 0].tolist()
        # random.shuffle(at_risk)
        # random.shuffle(control)
        # ...
        # test_set = processed.loc[test_indices]
        # train_set = processed.loc[train_indices]

        self.train_input: np.ndarray = np.asarray(list(train_input['PixelArray']))
        self.train_targets: np.ndarray = np.asarray(train_targets).astype('float32')

        # For GANs we need to normalize our pixels in the range [-1, 1]
        self.train_input = self.train_input * 2 - 1.

        if not KERAS_SEQUENTIAL:
            self.train_attrs: np.ndarray = np.asarray(list(train_input['MetaAttrs']))
            scaler = MinMaxScaler(copy=False)
            scaler.fit(self.train_attrs)
            scaler.transform(self.train_attrs)

        img_shape = self.train_input.shape[1:3]

        # Set some values
        self.params['n_train'] = self.train_input.shape[0]
        self.input_shape = img_shape
        self.train_input = self.train_input.reshape(
            (self.params['n_train'], img_shape[0], img_shape[1], self.n_channel)
        )

        if not KERAS_SEQUENTIAL:
            self.train_attrs = self.train_attrs.reshape(
                (self.params['n_train'], self.n_meta_attrs)
            )

        print(
            "Finished populating learning data:"
            f"\n\tImage shape:\t\t{img_shape}"
            f"\n\tTraining Input:\t\t{self.params['n_train']} images"
        )

        if save:
            # Pickle the four sets to disk
            print(f"Saving Numpy array learning data to {self.store_path}")
            np.save(self.processed_data_paths['train_input'], self.train_input)
            np.save(self.processed_data_paths['train_targets'], self.train_targets)

            if not KERAS_SEQUENTIAL:
                np.save(self.processed_data_paths['train_attrs'], self.train_attrs)
        return

    def get_train_batch(self, batch_size: int) -> tuple:
        """
        Load and grab 'batch_size' images + targets from our saved numpy arrays.
        This python generator format is used in the Keras fit function
        We can use the numpy memmap tools here to partially read the files
        """
        train_inputs = np.load(self.processed_data_paths['train_input'], mmap_mode='r+')
        train_targets = np.load(self.processed_data_paths['train_targets'], mmap_mode='r+')

        rng = np.random.default_rng(SEED)
        indices = np.arange(train_targets.shape[0])
        rng.shuffle(indices)

        # We'll shuffle the arrays and then save them again at the end of this generator
        # so on the next np.load we'll be working with last time's shuffled images
        train_inputs = train_inputs[indices]
        train_targets = train_targets[indices]

        if not KERAS_SEQUENTIAL:
            train_attrs = np.load(self.processed_data_paths['train_attrs'], mmap_mode='r+')
            train_attrs = train_attrs[indices]

        i_batch = 0
        start = 0
        num_batches = train_inputs.shape[0] // batch_size

        # The python generator bit. Yield batches of data.
        while True:
            end = start + batch_size
            if KERAS_SEQUENTIAL:
                yield train_inputs[start:end], train_targets[start:end]
            else:
                yield (train_inputs[start:end], train_attrs[start:end]), train_targets[start:end]
            start = end

            if i_batch < num_batches:
                i_batch += 1
            else:
                # Reshuffle the arrays for a new epoch
                i_batch = 0
                start = 0
                rng.shuffle(indices)
                train_inputs = train_inputs[indices]
                train_targets = train_targets[indices]
                if not KERAS_SEQUENTIAL:
                    train_attrs = train_attrs[indices]

    def get_real_img_batch(self, batch_size: int) -> tuple:
        """
        Load and grab 'batch_size' real images from our saved numpy arrays.
        This python generator format is used in the Keras fit function
        We can use the numpy memmap tools here to partially read the files
        """
        train_inputs = np.load(self.processed_data_paths['train_input'], mmap_mode='r+')

        rng = np.random.default_rng(SEED)
        indices = np.arange(train_inputs.shape[0])
        rng.shuffle(indices)

        # We'll shuffle the arrays and then save them again at the end of this generator
        # so on the next np.load we'll be working with last time's shuffled images
        train_inputs = train_inputs[indices]

        if not KERAS_SEQUENTIAL:
            train_attrs = np.load(self.processed_data_paths['train_attrs'], mmap_mode='r+')
            train_attrs = train_attrs[indices]

        i_batch = 0
        start = 0
        num_batches = train_inputs.shape[0] // batch_size

        # The python generator bit. Yield batches of data.
        while True:
            end = start + batch_size
            if KERAS_SEQUENTIAL:
                yield train_inputs[start:end]
            else:
                yield (train_inputs[start:end], train_attrs[start:end])
            start = end

            if i_batch < num_batches:
                i_batch += 1
            else:
                # Reshuffle the arrays for a new epoch
                i_batch = 0
                start = 0
                rng.shuffle(indices)
                train_inputs = train_inputs[indices]
                if not KERAS_SEQUENTIAL:
                    train_attrs = train_attrs[indices]

    def clear_array_attributes(self):
        self.train_input = None
        self.train_targets = None
        self.test_input = None
        self.test_targets = None
        self.train_attrs = None
        self.test_attrs = None

    def clear_dataframe(self):
        self.dataframe = None
        self._loaded_df = False

    def teardown(self):
        """
        Various teardown operations. Important for memory cleanup
        """
        # First let's delete all the processed images
        processed_data_path = self.store_path + "/processed_data"
        if os.path.exists(processed_data_path):
            print(f"Teardown: Deleting directory {processed_data_path}")
            shutil.rmtree(processed_data_path)

        self.clear_array_attributes()
        self.clear_dataframe()

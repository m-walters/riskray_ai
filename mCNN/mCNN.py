import os
import shutil

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Have to create this as a packaged pyproject to use "riskray.ai.dicom..."
from .. dicom.dataframes import TrainingDataFrame
from .. dicom.dataset_tools import center_crop_df, pad_df, downscale, resize, META_DATA_KEYS
from .. dicom.image_plotter import plot_grid_df
from .. settings import STORE_PATH, KERAS_SEQUENTIAL, SEED


def load_keras_model(path):
    from keras.models import load_model
    return load_model(path)


class MixedCNNTrainer:
    """
    Trainer for our Mixed CNN
    Takes in a TrainingDataFrame instance on init.

    Example:
        df = DefaultDataFrame()
        mlp = MixedCNNTrainer(df)
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
    # Some MLP parameters
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

    def export_prediction_images(self, start=0, num=None, nn_model=None):
        if self.train_input is None and self.test_input is None:
            # Populate the data
            self.populate_learning_data()
        # Create image path if needed
        if not os.path.exists(self.img_path):
            print(f"Creating Image Store path {self.img_path}")
            os.makedirs(self.img_path)

        images = {
            "train_control": [],
            "train_at_risk": [],
            "test_control": [],
            "test_at_risk": []
        }
        for img, target in zip(self.train_input, self.train_targets):
            if target == 0:
                images['train_control'].append(img)
            else:
                images['train_at_risk'].append(img)
        for img, target in zip(self.test_input, self.test_targets):
            if target == 0:
                images['test_control'].append(img)
            else:
                images['test_at_risk'].append(img)

        all_images = np.asarray(images['train_control'])
        all_images = np.append(all_images, np.asarray(images['train_at_risk']), axis=0)
        all_images = np.append(all_images, np.asarray(images['test_control']), axis=0)
        all_images = np.append(all_images, np.asarray(images['test_at_risk']), axis=0)
        tmp_df = pd.DataFrame({'PixelArray': [_ for _ in all_images]})
        predictions = None
        if nn_model:
            predictions = nn_model(all_images)
            predictions = predictions.numpy().flatten()

        counts = [
            (len(images['train_control']), "Trn-Ctrl"),
            (len(images['train_at_risk']), "Trn-Risk"),
            (len(images['test_control']), "Test-Ctrl"),
            (len(images['test_at_risk']), "Test-Risk"),
        ]
        all_labels = []
        i_img = 0
        for count in counts:
            n_img, lbl_base = count
            # Modify lbl
            for i in range(n_img):
                if nn_model and predictions is not None:
                    i_pred = i_img + i
                    lbl = f"{lbl_base} | {100 * predictions[i_pred]:.1f}%"
                    all_labels.append(lbl)
            i_img += n_img

        tmp_df['Annotation'] = all_labels

        # Plot the control images
        plot_grid_df(
            tmp_df,
            nrow=4,
            ncol=4,
            start=start,
            num=num,
            save=True,
            save_path=self.img_path + "/prediction.jpg",
            annotate=True,
        )

        del tmp_df, all_images, counts
        return

    def export_dataframe_images(self, start=0, num=None, nn_model=None):
        """
        Output images of the processed images for inspection

        :param start:       Dataframe index to start images from
        :param num:         Limit number output with num
        :parma nn_model:    Include a NN model
        """
        # Create image path if needed
        if not os.path.exists(self.img_path):
            print(f"Creating Image Store path {self.img_path}")
            os.makedirs(self.img_path)

        # Repeat the processing steps from the raw dataframe images
        if self.dataframe.dataframe is None:
            self.load_dataframe()

        # Cropping, padding, downscaling
        processed = center_crop_df(self.dataframe.dataframe, 'PixelArray', self.image_reshape[0], self.image_reshape[1])
        processed = pad_df(processed, 'PixelArray', self.image_reshape[0], self.image_reshape[1])
        # processed = downscale(processed, 'PixelArray', self.downscale_factor)
        outshape = tuple([i // self.downscale_factor for i in self.image_reshape])
        processed = resize(processed, 'PixelArray', outshape=outshape)

        if nn_model:
            # Generate predictions and add to dataframe
            # The added frills here just shape things for model input
            images = np.expand_dims(np.stack(processed['PixelArray'].values), -1)
            predictions = nn_model(images)
            processed['Predictions'] = predictions.numpy().flatten()

        # Plot the control images
        plot_grid_df(
            processed[processed['Target'] == 0],
            nrow=4,
            ncol=4,
            start=start,
            num=num,
            save=True,
            save_path=self.img_path + "/control.jpg",
            annotate=True,
        )

        # Plot the at-risk images
        plot_grid_df(
            processed[processed['Target'] == 1],
            nrow=4,
            ncol=4,
            start=start,
            num=num,
            save=True,
            save_path=self.img_path + "/at-risk.jpg",
            annotate=True
        )

        del processed
        return

    def populate_learning_data(self, save=True, cleanup=True, keras_sequential=KERAS_SEQUENTIAL):
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

        train_input, test_input, train_targets, test_targets = train_test_split(
            processed[['PixelArray', 'MetaAttrs']],
            processed['Target'],
            test_size=self.params['test_train_ratio'],
            random_state=SEED
        )

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
        self.test_input: np.ndarray = np.asarray(list(test_input['PixelArray']))
        self.train_targets: np.ndarray = np.asarray(train_targets).astype('float32')
        self.test_targets: np.ndarray = np.asarray(test_targets).astype('float32')

        if not keras_sequential:
            self.train_attrs: np.ndarray = np.asarray(list(train_input['MetaAttrs']))
            self.test_attrs: np.ndarray = np.asarray(list(test_input['MetaAttrs']))
            scaler = MinMaxScaler(copy=False)
            scaler.fit(self.train_attrs)
            scaler.transform(self.train_attrs)
            scaler.transform(self.test_attrs)

        img_shape = self.train_input.shape[1:3]
        if img_shape != self.test_input.shape[1:3]:
            raise Exception(
                "Training images and Test images have mismatching XY sizes: "
                f"{img_shape} AND {self.test_input.shape[1:3]}"
            )

        # Set some values
        self.params['n_train'] = self.train_input.shape[0]
        self.params['n_validate'] = int((self.train_input.shape[0] * self.params['validation_split']))
        self.params['n_test'] = self.test_input.shape[0]
        self.input_shape = img_shape
        self.train_input = self.train_input.reshape(
            (self.params['n_train'], img_shape[0], img_shape[1], self.n_channel)
        )
        self.test_input = self.test_input.reshape(
            (self.params['n_test'], img_shape[0], img_shape[1], self.n_channel)
        )
        
        if not keras_sequential:
            self.train_attrs = self.train_attrs.reshape(
                (self.params['n_train'], self.n_meta_attrs)
            )
            self.test_attrs = self.test_attrs.reshape(
                (self.params['n_test'], self.n_meta_attrs)
            )

        print(
            "Finished populating learning data:"
            f"\n\tImage shape:\t\t{img_shape}"
            f"\n\tTraining Input:\t\t{self.params['n_train']} images ({self.params['n_validate']} Validation)"
            f"\n\tTraining Num At-Risk:\t{self.train_targets.sum()} ({(self.train_targets.mean() * 100):.1f}% of set)"
            f"\n\tTest Input:\t\t{self.params['n_test']} images"
            f"\n\tTest Num At-Risk:\t\t{self.test_targets.sum()} ({(self.test_targets.mean() * 100):.1f}% of set)"
        )

        if save:
            # Pickle the four sets to disk
            print(f"Saving Numpy array learning data to {self.store_path}")
            np.save(self.processed_data_paths['train_input'], self.train_input)
            np.save(self.processed_data_paths['test_input'], self.test_input)
            np.save(self.processed_data_paths['train_targets'], self.train_targets)
            np.save(self.processed_data_paths['test_targets'], self.test_targets)

            if not keras_sequential:
                np.save(self.processed_data_paths['train_attrs'], self.train_attrs)
                np.save(self.processed_data_paths['test_attrs'], self.test_attrs)

        return

    def train_generator(self, batch_size: int) -> tuple:
        """
        Load and grab 'batch_size' images + targets from our saved numpy arrays.
        This generator format is used in the Keras fit function
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

    def test_generator(self, batch_size: int) -> tuple:
        """
        Load and grab 'batch_size' images + targets from our saved numpy arrays.
        This generator format is used in the Keras fit function
        We can use the numpy memmap tools here to partially read the files
        """
        test_inputs = np.load(self.processed_data_paths['test_input'], mmap_mode='r+')
        test_targets = np.load(self.processed_data_paths['test_targets'], mmap_mode='r+')

        rng = np.random.default_rng(SEED)
        indices = np.arange(test_targets.shape[0])
        rng.shuffle(indices)

        # We'll shuffle the arrays and then save them again at the end of this generator
        # so on the next np.load we'll be working with last time's shuffled images
        test_inputs = test_inputs[indices]
        test_targets = test_targets[indices]
        if not KERAS_SEQUENTIAL:
            test_attrs = np.load(self.processed_data_paths['test_attrs'], mmap_mode='r+')
            test_attrs = test_attrs[indices]

        i_batch = 0
        start = 0
        num_batches = test_inputs.shape[0] // batch_size
        while True:
            end = start + batch_size
            if KERAS_SEQUENTIAL:
                yield test_inputs[start:end], test_targets[start:end]
            else:
                yield (test_inputs[start:end], test_attrs[start:end]), test_targets[start:end]
            start = end

            if i_batch < num_batches:
                i_batch += 1
            else:
                # Reshuffle the arrays
                i_batch = 0
                start = 0
                rng.shuffle(indices)
                test_inputs = test_inputs[indices]
                test_targets = test_targets[indices]
                if not KERAS_SEQUENTIAL:
                    test_attrs = test_attrs[indices]

    def set_array_attributes_from_disk(self):
        """
        From the npy files created for generator use, re-set our learning array attributes
        in case they have been cleared to save memory, for instance
        """
        self.train_input = np.load(self.processed_data_paths['train_input'])
        self.test_input = np.load(self.processed_data_paths['test_input'])
        self.train_targets = np.load(self.processed_data_paths['train_targets'])
        self.test_targets = np.load(self.processed_data_paths['test_targets'])
        if not KERAS_SEQUENTIAL:
            self.train_attrs = np.load(self.processed_data_paths['train_attrs'])
            self.test_attrs = np.load(self.processed_data_paths['test_attrs'])

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

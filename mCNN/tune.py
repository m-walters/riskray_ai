import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional, TypedDict

import numpy as np
import ray
import yaml
from keras import Model, optimizers, Sequential
from keras.layers import (
    BatchNormalization, Concatenate, Conv2D, Dense, Dropout,
    Flatten, Input, LeakyReLU, MaxPooling2D
)
from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune import CLIReporter, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper
from sklearn import metrics as skmetrics

from ..dicom.dataframes import DefaultDataFrame
from ..mCNN.mCNN import MixedCNNTrainer
from ..settings import KERAS_SEQUENTIAL, STORE_PATH, USE_GENERATORS, USE_GPU


# Use matplotlib as backend, no gui -- Or set MPLBACKEND env var
# matplotlib.use('agg')


class TuneConfig(TypedDict):
    # Identify the run
    experiment_name: str
    # Number epochs per run
    num_training_iterations: int
    # Number of repeated runs per hyper-param point
    # This will run num_samples for EACH item in your grid_searches
    num_samples: int
    # [min, max] learning rates to sample from. [1e-5, 1e-1] are good. 1e-4 too.
    learning_rate_range: List[float]
    # Range [0,1]. Set aside this ratio of images for final testing
    train_test_split: float
    # CPUs/GPUs allocated to the entire tune. Default is 4 CPU, 0 GPU
    allocated_cpus: Optional[int]
    allocated_gpus: Optional[int]
    # Resources per trial. Can also be fractions.
    # If non-null, should take form {"CPU": <int/float>, "GPU": <int/float>}
    resources_per_trial: Optional[Dict[str, int]]


def risk_ray_mixed_data(config, trainer):
    """
    Return our Keras Mixed-Data model
    This model combines the regular image recognition network, and then merges (concatenates) it with a
    small network that processes other parameters such as keV, age, etc.
    Considerations: We may want to do a second CNN after a modified
    """

    # Image network
    img_input = Input((trainer.input_shape[0], trainer.input_shape[1], trainer.n_channel))
    img_net = Conv2D(
        config['filters_0'],  # Output filters
        config['kernel_0'],  # Some odd number
        padding=config['padding'],
        strides=config['strides']
    )(img_input)
    img_net = LeakyReLU(alpha=0.2)(img_net)
    # img_net = Dropout(config['dropout'])(img_net)
    img_net = MaxPooling2D(padding='same')(img_net),
    img_net = Conv2D(
        config['filters_1'],
        config['kernel_1'],
        padding=config['padding'],
        strides=config['strides']
    )(img_net)
    img_net = LeakyReLU(alpha=0.2)(img_net)
    img_net = Flatten()(img_net)
    img_net = Dense(config['dense_0'])(img_net)
    img_net = LeakyReLU(alpha=0.2)(img_net)

    # Now our parameter network
    param_input = Input((trainer.n_meta_attrs,))
    # Let's give it tanh for some -1 values
    param_net = Dense(16, activation='tanh')(param_input)

    merged = Concatenate()([img_net, param_net])

    merged = Dense(16, activation='relu')(merged)
    # merged = Dense(8, activation='relu')(merged)
    # merged = BatchNormalization()(merged)
    output = Dense(1, activation="sigmoid")(merged)

    return Model(inputs=(img_input, param_input), outputs=output)


def risk_ray_sequential(config, trainer):
    """
    Return our Keras Sequential model
    """
    return Sequential(
        [
            Conv2D(
                config['filters_0'],  # Output filters
                config['kernel_0'],  # Some odd number
                input_shape=(trainer.input_shape[0], trainer.input_shape[1], trainer.n_channel),
                padding=config['padding'],
                strides=config['strides']
            ),
            Dropout(config['dropout']),
            LeakyReLU(alpha=0.2),
            # MaxPooling2D(padding='same'),
            # Conv2D(16, 15, padding='same'),
            # Activation('relu'),
            # MaxPooling2D(padding='same'),
            Conv2D(
                config['filters_1'],
                config['kernel_1'],
                padding=config['padding'],
                strides=config['strides']
            ),
            Flatten(),
            Dense(
                config['dense_0'],
                activation="relu"
            ),
            BatchNormalization(),
            Dense(1, activation="sigmoid")
        ]
    )


def train_mlp(config):
    """
    Train the MLP network

    config = {
      # Related to our internal Trainer data prep
      'image_reshape': ...,
      'downscale_factor': ...,
      'experiment_name': ...,

      # Other parameters
      'epochs': ...,
      'learning_rate': ...,
      'filters_0': ...,
      'kernel_0': ...,
      'dropout': ...,
      'kernel_1': ...,
      'dense_0': ...,
    }
    """

    # Setup data and Trainer
    image_reshape = config['image_reshape']
    downscale_factor = config['downscale_factor']
    try:
        riskray_run_name = f"{config['experiment_name']}/{ray.train.get_context().get_trial_name()}"
    except Exception:
        riskray_run_name = f"{config['experiment_name']}/testing"
    run_dir = f"{STORE_PATH}/experiments/{riskray_run_name}"

    df = DefaultDataFrame()
    trainer = MixedCNNTrainer(df, run=riskray_run_name, downscale_factor=downscale_factor, image_reshape=image_reshape)

    trainer.populate_learning_data()
    if USE_GENERATORS:
        # Since we are using generators, we'll generate and save the numpy files and access those via the generators
        # So we needn't keep the other dataframes and arrays in local memory after populating
        trainer.clear_array_attributes()
        # trainer.clear_dataframe()

    # TODO -- Conv2D assumes channels_last for data_format. Verify this is the case in our image batches

    epochs = config['epochs']

    if KERAS_SEQUENTIAL:
        model = risk_ray_sequential(config, trainer)
    else:
        model = risk_ray_mixed_data(config, trainer)

    model.compile(
        optimizer=optimizers.SGD(learning_rate=config['learning_rate']),
        loss=['binary_crossentropy'],
        metrics=['accuracy']
    )

    # Write the model summary
    with open(run_dir + "model_summary.txt", 'w') as mf:
        mf.write(model.summary())

    if USE_GENERATORS:
        # Grab training generators
        batch_size = config['batch_size']
        train_generator = trainer.train_generator(batch_size)
        train_batch_per_epoch = int(trainer.params['n_train'] // batch_size)
        test_generator = trainer.test_generator(batch_size)
        val_batch_per_epoch = int(trainer.params['n_validate'] // batch_size)
        test_batch_per_epoch = int(trainer.params['n_test'] // batch_size)
        # Call the training function
        model.fit(
            train_generator,
            batch_size=batch_size,
            steps_per_epoch=train_batch_per_epoch,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=val_batch_per_epoch,
            verbose=0,
            callbacks=[ReportCheckpointCallback()]
        )

    else:
        if not KERAS_SEQUENTIAL:
            raise Exception("Not configured for Non-generator with Mixed data types")
        # Call the training function
        model.fit(
            trainer.train_input,
            trainer.train_targets,
            # validation_split=trainer.params['validation_split'],
            validation_data=(trainer.test_input, trainer.test_targets),
            epochs=epochs,
            verbose=0,
            callbacks=[ReportCheckpointCallback()]
            # callbacks=[ReportCheckpointCallback(
            #     {
            #         "mean_accuracy": "accuracy",
            #         "loss": "loss",
            #         "validation_accuracy": "val_accuracy",
            #         "validation_loss": "val_loss",
            #     }
            # )]
        )

    # Save our model
    model.save(f"{run_dir}/models/model.keras", save_format="keras")

    # Let's get the f1 score, precision, recall, and support
    if USE_GENERATORS:
        # Let's quickly reload the test arrays directly from paths since the trainer attributes were cleared earlier
        test_input = np.load(trainer.processed_data_paths['test_input'], 'r')
        if not KERAS_SEQUENTIAL:
            test_attrs = np.load(trainer.processed_data_paths['test_attrs'], 'r')
        else:
            test_attrs = None
        test_targets = np.load(trainer.processed_data_paths['test_targets'], 'r')
    else:
        test_input = trainer.test_input
        test_attrs = trainer.test_attrs
        test_targets = trainer.test_targets

    if not KERAS_SEQUENTIAL:
        y_pred = model.predict(x=[test_input, test_attrs])
    else:
        y_pred = model.predict(x=test_input)

    # Print a classification report and save it
    print("<><><><><><><   Classification Report   ><><><><><><>")
    # For now threshold 0.5. We probably won't keep this one around though, or use best thresh.
    pred_bool = np.round(y_pred)
    report = skmetrics.classification_report(test_targets, pred_bool)
    print(report)
    with open(f"{run_dir}/class_report.txt", 'w') as f:
        f.write(report)

    teardown(trainer)


def teardown(riskray_trainer):
    riskray_trainer.teardown()
    # Remove this additional directory
    store_path = riskray_trainer.store_path
    if os.path.exists(store_path):
        print(f"Deleting store path: {store_path}")
        shutil.rmtree(store_path)


def trial_name_string(trial):
    # trial is a Trial object
    return str(trial)


def tune_mlp(config: TuneConfig, interactive: bool) -> ResultGrid:
    """
    Fire off our trials

    A note on choosing optimizers/schedulers:
        https://docs.ray.io/en/master/tune/tutorials/overview.html#id13

    :param config:      TuneConfig dict
    :param interactive: Whether running in interactive mode
    """
    experiment_name = config["experiment_name"]

    # Clear out stale tune runs
    result_path = f"{STORE_PATH}/experiments/{experiment_name}"
    model_save_path = f"{result_path}/models"
    training_data_path = f"{STORE_PATH}/training/default/{experiment_name}"

    if interactive:
        print(
            "You are about to replace the contents of the following directories:"
            f"\n\t{result_path}\n\t{model_save_path}\n\t{training_data_path}"
        )
        answer = input("Continue? [y/n] ")
        if answer != 'y':
            # Let's be strict here and abort
            print("Aborting.")
            sys.exit(0)
    else:
        print(
            "Replacing contents of the following directories:"
            f"\n\t{result_path}\n\t{model_save_path}\n\t{training_data_path}"
        )

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
        os.mkdir(result_path)
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
        os.mkdir(model_save_path)
    if os.path.exists(training_data_path):
        shutil.rmtree(training_data_path)
        os.mkdir(training_data_path)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    stopper = TrialPlateauStopper(
        # metric="mean_accuracy",
        # metric="accuracy",
        # metric_threshold=0.85,
        # mode="min",
        metric="loss",
        std=0.01,
        num_results=20,
        grace_period=10
    )

    resources_per_trial = config.get("resources_per_trial", {"CPU": 1, "GPU": 0})
    scaling_config = ScalingConfig(
        use_gpu=USE_GPU,
        resources_per_worker=resources_per_trial
    )
    run_config_params = {
        # Resources
        "scaling_config": scaling_config,
        # Preprocessing params
        "experiment_name": config["experiment_name"],
        # 'image_reshape': (800,700),
        'image_reshape': (1000, 800),
        'batch_size': 10,
        'downscale_factor': tune.choice([4, 8]),
        # These are integers / floats
        'epochs': config["num_training_iterations"],
        'learning_rate': tune.loguniform(
            config["learning_rate_range"][0],
            config["learning_rate_range"][1],
        ),
        # 'learning_rate': 0.01,
        'dropout': tune.uniform(0.1, 0.4),
        'padding': "same",
        'strides': 4,
        'filters_0': tune.choice([8, 16, 32]),
        'filters_1': tune.choice([32, 64]),
        'kernel_0': tune.choice([9, 15, 21]),
        'kernel_1': tune.choice([9, 15, 21]),
        # Note, MixedCNN merges with attrs at a size of ~16.
        # So you probably want your dense_0 to be similar
        'dense_0': tune.choice([8, 16, 32]),
    }

    # Iterate over the tune config file and overwrite any provided params from it
    for key in run_config_params:
        if key in config:
            run_config_params[key] = config[key]

    # TODO -- Can we optimize for the product of train_accuracy x val_accuracy?
    search_alg = HyperOptSearch(metric="accuracy", mode="max")

    tune_config = tune.TuneConfig(
        scheduler=sched,
        metric="loss",
        mode="min",
        search_alg=search_alg,
        num_samples=config["num_samples"],
        trial_dirname_creator=trial_name_string,
        trial_name_creator=trial_name_string,
        reuse_actors=False,
    )
    # This got deprecated in newer version, using new AIR formatting?
    # To re-instate, use legacy by setting env var RAY_AIR_NEW_OUTPUT=0
    progress_reporter = CLIReporter(
        max_report_frequency=60,  # seconds between reports
    )
    run_config = RunConfig(
        name=experiment_name,
        stop=stopper,
        progress_reporter=progress_reporter,
    )
    tuner = tune.Tuner(
        train_mlp,
        param_space=run_config_params,
        tune_config=tune_config,
        run_config=run_config,
    )

    try:
        return tuner.fit()
    except RuntimeError as RE:
        if RE.__str__().startswith(
                "Trying to sample a configuration from HyperOptSearch, but no search space has been defined."
        ):
            # Our run_config_params were all constants
            tune_config.search_alg = None
            tuner = tune.Tuner(
                train_mlp,
                param_space=run_config_params,
                tune_config=tune_config,
                run_config=run_config,
            )
            return tuner.fit()


def validate_config(config: TuneConfig) -> None:
    if not isinstance(config["experiment_name"], str) or len(config["experiment_name"]) == 0:
        raise ValueError("Non-empty experiment_name required")
    if not isinstance(config["num_training_iterations"], int):
        raise ValueError("num_training_iterations integer required")
    if not isinstance(config["num_samples"], int):
        raise ValueError("num_samples integer required")
    lr_range = config["learning_rate_range"]
    if not isinstance(lr_range[0], float) or not isinstance(lr_range[1], float):
        raise ValueError("learning_rate_range must be list of two floats")
    tt_split = config["train_test_split"]
    if not isinstance(tt_split, float) or not 0 <= tt_split <= 1:
        raise ValueError("train_test_split must be float in range [0,1]")
    resources_per_trial = config.get("resources_per_trial", {})
    if not USE_GPU and resources_per_trial.get("gpu", 0) > 0:
        raise ValueError("resources_per_trial requesting non-zero GPUs. Set USE_GPU environment variable to true")
    if not USE_GPU and config.get("allocated_gpus", 0) > 0:
        raise ValueError("Requesting non-zero GPUs for ray tune allocation. Set USE_GPU environment variable to true")


if __name__ == "__main__":
    # TODO:
    #   1. Random Seed management for tensorflow, raytune, etc.
    #   2. Can we optimize for the product of train_accuracy x val_accuracy?

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="RayTune experiment config file. See TuneConfig in training/tune.py"
    )

    parser.add_argument(
        "--non-interactive",
        default=False,
        action="store_true",
        help="Run in non-interactive mode"
    )

    args, _ = parser.parse_known_args()

    # Parse the raytune config
    with open(args.config, 'r') as f:
        try:
            config: TuneConfig = yaml.safe_load(f)
        except yaml.YAMLError as YE:
            print(f"YAML parsing error of tune config:\n\t{YE}")
            sys.exit()

    # Validate config
    validate_config(config)

    interactive = not args.non_interactive

    # Init RayTune
    allocated_cpu = config.get("allocated_cpus", 4)
    allocated_gpu = config.get("allocated_gpus", 0)
    ray.init(num_cpus=allocated_cpu, num_gpus=allocated_gpu)

    # Perform the experiment
    result: ResultGrid = tune_mlp(config, interactive)

    # Various attributes from analysis
    best_trial = result.get_best_result()
    # best_result = result.best_result  # Get best trial's last results
    # best_result_df = result.get_dataframe()  # Get best result as pandas dataframe

    print(
        "-----------------  END OF RUN BEST RESULTS  -----------------\n"
        f"LOCATION \n{best_trial.path}\n"
        f"CONFIG \n{best_trial.config}\n"
        f"OPTIMUM CHECKPOINTS \n{[(cp.path, metric['accuracy']) for cp, metric in best_trial.best_checkpoints]}\n"
    )

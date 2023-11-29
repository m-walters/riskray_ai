import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional, TypedDict

import keras.losses
import numpy as np
import ray
import yaml

from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune import CLIReporter, ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper
from sklearn.metrics import classification_report
from keras import Model, Sequential, optimizers

from ..dicom.dataframes import DefaultDataFrame
from ..settings import KERAS_SEQUENTIAL, STORE_PATH, USE_GENERATORS, USE_GPU
from ..cDCGAN.cDCGAN import (
    GeneratorConfig, DiscriminatorConfig, GANTrainer, GANMonitor, cDCGAN,
    risk_ray_gan_generator, risk_ray_gan_discriminator
)


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
    # CPUs/GPUs allocated to the entire tune. Default is 4 CPU, 0 GPU
    allocated_cpus: Optional[int]
    allocated_gpus: Optional[int]
    # Resources per trial. Can also be fractions.
    # If non-null, should take form {"CPU": <int/float>, "GPU": <int/float>}
    resources_per_trial: Optional[Dict[str, int]]

    # GAN-related configs
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig


def train_gan(config):
    """
    Train the GAN
    """
    # Setup data and Trainer
    image_reshape = config['image_reshape']
    downscale_factor = config['downscale_factor']
    try:
        riskray_run_name = f"{config['experiment_name']}/{ray.train.get_context().get_trial_name()}"
    except Exception:
        riskray_run_name = f"{config['experiment_name']}/testing"

    df = DefaultDataFrame()
    trainer = GANTrainer(df, run=riskray_run_name, downscale_factor=downscale_factor, image_reshape=image_reshape)

    trainer.populate_learning_data()
    if USE_GENERATORS:
        # Since we are using generators, we'll generate and save the numpy files and access those via the generators
        # So we needn't keep the other dataframes and arrays in local memory after populating
        trainer.clear_array_attributes()
        # trainer.clear_dataframe()

    # TODO -- Conv2D assumes channels_last for data_format. Verify this is the case in our image batches

    epochs = config['epochs']

    # GAN
    gen_config = {
        "latent_edge": config['latent_edge'],
        "latent_channels": config['latent_channels'],
        "num_classes": config['num_classes'],
        **config['generator']
    }
    generator: Sequential = risk_ray_gan_generator(gen_config)
    discriminator: Sequential = risk_ray_gan_discriminator(config['discriminator'], trainer)
    # Latent dimensions
    # TODO -- f"MW Conditional Upgrade -- add num_classes
    # latent_dim = int(config['latent_channels'] + config['num_classes'])
    latent_dim = config['latent_channels']

    # Initialize the GAN and compile
    gan = cDCGAN(discriminator, generator, latent_dim)
    adam_config = config.get("adam", None)
    if adam_config:
        d_optimizer = optimizers.Adam(**adam_config)
        g_optimizer = optimizers.Adam(**adam_config)
    else:
        d_optimizer = optimizers.SGD(config['learning_rate'])
        g_optimizer = optimizers.SGD(config['learning_rate'])

    gan.compile(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    # Grab training generators
    batch_size = config['batch_size']
    # train_generator = trainer.get_train_batch(batch_size)
    real_image_batch = trainer.get_real_img_batch(batch_size)
    train_batch_per_epoch = int(trainer.params['n_train'] // batch_size)
    # test_generator = trainer.test_generator(batch_size)
    # val_batch_per_epoch = int(trainer.params['n_validate'] // batch_size)
    # test_batch_per_epoch = int(trainer.params['n_test'] // batch_size)

    # Call the training function
    gan.fit(
        real_image_batch,
        batch_size=batch_size,
        steps_per_epoch=train_batch_per_epoch,
        epochs=epochs,
        verbose=0,
        callbacks = [GANMonitor(riskray_run_name, real_image_batch, num_img=2, latent_dim=latent_dim)]
    )

    # model_dir = f"{STORE_PATH}/experiments/models/{riskray_run_name}"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # Save our model
    # gan.save(model_dir, save_format='tf')
    #
    # print("<><><><><><><   Classification Report   ><><><><><><>")
    # # Let's get the f1 score, precision, recall, and support
    # if USE_GENERATORS:
    #     # Let's quickly reload the test arrays directly from paths since the trainer attributes were cleared earlier
    #     imgs = np.load(trainer.processed_data_paths['test_input'], 'r')
    #     attrs = np.load(trainer.processed_data_paths['test_attrs'], 'r')
    #     test_targets = np.load(trainer.processed_data_paths['test_targets'], 'r')
    #     # y_pred = model.predict(
    #     #     test_generator,
    #     #     batch_size=batch_size,
    #     #     steps=test_batch_per_epoch
    #     # )
    #     y_pred = model.predict(
    #         x=[imgs, attrs]
    #     )
    #     y_pred_bool = np.round(y_pred)
    #     report = classification_report(test_targets, y_pred_bool)
    # else:
    #     y_pred = model(trainer.test_input, training=False)
    #     y_pred_bool = np.round(y_pred)
    #     report = classification_report(trainer.test_targets, y_pred_bool)
    #
    # print(report)
    # with open(f"{model_dir}/scores.txt", 'w') as f:
    #     f.write(report)
    # # test_loss, test_acc = model.evaluate(trainer.test_input, trainer.test_targets, verbose=0)
    # #
    # # print(test_acc)
    # # print(test_loss)

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


def tune_gan(config: TuneConfig, interactive: bool) -> ResultGrid:
    """
    Fire off our trials

    A note on choosing optimizers/schedulers:
        https://docs.ray.io/en/master/tune/tutorials/overview.html#id13

    :param config:      TuneConfig dict
    :param interactive: Whether running in interactive mode
    """
    experiment_name = config["experiment_name"]

    # Clear out stale tune runs
    result_path = f"{STORE_PATH}/training/experiments/{experiment_name}"
    model_save_path = f"{STORE_PATH}/training/models/{experiment_name}"
    training_data_path = f"{STORE_PATH}/training/default/{experiment_name}"
    generated_image_path = f"{STORE_PATH}/generated_images/{experiment_name}"

    if interactive:
        print("You are about to replace the contents of the following directories:"
              f"\n\t{result_path}\n\t{model_save_path}\n\t{training_data_path}\n\t{generated_image_path}")
        answer = input("Continue? [y/n] ")
        if answer != 'y':
            # Let's be strict here and abort
            print("Aborting.")
            sys.exit(0)
    else:
        print("Replacing contents of the following directories:"
              f"\n\t{result_path}\n\t{model_save_path}\n\t{training_data_path}")

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
        os.mkdir(result_path)
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
        os.mkdir(model_save_path)
    if os.path.exists(training_data_path):
        shutil.rmtree(training_data_path)
        os.mkdir(training_data_path)
    if os.path.exists(generated_image_path):
        shutil.rmtree(generated_image_path)
        os.mkdir(generated_image_path)


    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)

    # stopper = TrialPlateauStopper(
    #     metric="combined_loss",
    #     num_results=5,
    #     grace_period=8
    # )

    resources_per_trial = config.get("resources_per_trial", {"CPU": 1, "GPU": 0})
    scaling_config = ScalingConfig(
        use_gpu=USE_GPU,
        resources_per_worker=resources_per_trial
    )
    # See here for making dependant/complex search spaces:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html#how-to-use-custom-and-conditional-search-spaces-in-tune
    run_config_params = {
        # Resources
        "scaling_config": scaling_config,
        # Preprocessing params
        "experiment_name": config["experiment_name"],
        # "image_reshape": (1000, 800),
        "image_reshape": (800, 800),
        "batch_size": 10,
        "downscale_factor": 10,
        "epochs": config["num_training_iterations"],
        
        # Learning configs
        # "learning_rate": tune.loguniform(
        #     config["learning_rate_range"][0],
        #     config["learning_rate_range"][1],
        # ),
        "adam": {
            "learning_rate": tune.loguniform(1e-3, 0.1),
        },
        
        # GAN configs -- Fix these for now
        "latent_edge": 20,
        "latent_channels": tune.choice([16, 32]),
        "num_classes": 2,
        "generator": {
            "kernel": tune.choice([9]),
            "out_0": tune.choice([64, 128]),
            "out_1": tune.choice([128, 256]),
            # "out_2": tune.choice([32]),
            "final_kernel": tune.choice([5])
        },
        "discriminator": {
            "filters_0": tune.choice([8, 16]),
            "kernel_0": tune.choice([9]),
            "filters_1": tune.choice([16, 32]),
            "kernel_1": tune.choice([5]),
        },
    }
    # Iterate over the tune config file and overwrite any provided params from it
    for key in run_config_params:
        if key in config:
            if key in ["generator", "discriminator"]:
                for gkey, val in config[key].items():
                    run_config_params[key][gkey] = val
            else:
                run_config_params[key] = config[key]

    # Set search_alg to None if wanting to do "grid_search"
    search_alg = HyperOptSearch(metric="d_loss", mode="min")
    tune_config = tune.TuneConfig(
        scheduler=sched,
        metric="d_loss",
        mode="min",
        search_alg=search_alg,
        num_samples=config["num_samples"],
        # num_samples=1, # For grid_search
        trial_dirname_creator=trial_name_string,
        trial_name_creator=trial_name_string,
        # Workaround for a reset_config bug
        reuse_actors=False,
    )
    # This got deprecated in newer version, using new AIR formatting?
    # To re-instate, use legacy by setting env var RAY_AIR_NEW_OUTPUT=0
    # progress_reporter = CLIReporter(
    #     metric_columns=["g_loss", "d_loss", "combined_loss", "gen_sample_accuracy", "real_sample_accuracy"],
    #     max_report_frequency=60,  # seconds between reports
    # )
    run_config = RunConfig(
        name=experiment_name,
        # stop=stopper,
        # progress_reporter=progress_reporter,
    )
    tuner = tune.Tuner(
        train_gan,
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
                train_gan,
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
        help="RayTune experiment config file. See TuneConfig in training/tune.py")

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
            config: TuneConfig  = yaml.safe_load(f)
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
    result: ResultGrid = tune_gan(config, interactive)

    # Various attributes from analysis
    best_trial = result.get_best_result()
    # best_result = result.best_result  # Get best trial's last results
    # best_result_df = result.get_dataframe()  # Get best result as pandas dataframe

    print(
        "-----------------  END OF RUN BEST RESULTS  -----------------\n"
        f"LOCATION \n{best_trial.path}\n"
        f"CONFIG \n{best_trial.config}\n"
        f"OPTIMUM CHECKPOINTS \n{' | '.join([cp.path for cp, metric in best_trial.best_checkpoints])}\n"
    )

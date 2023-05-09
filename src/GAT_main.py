import argparse


from utils import PPI_NUM_INPUT_FEATURES, PPI_NUM_CLASSES, BEST_VAL_MICRO_F1,BEST_VAL_LOSS,PATIENCE_CNT,DEVICE
from GAT_train import *
from GAT_model import *
import time

device = torch.device(DEVICE)
def train_gat_ppi(config):
    """
    Very similar to Cora's training script. The main differences are:
    1. Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    2. Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    3. Model architecture and hyperparams are a bit different (as reported in the GAT paper)

    """
    global BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT

    BEST_VAL_MICRO_F1 = 0
    BEST_VAL_LOSS = 0
    PATIENCE_CNT = 0

    # Checking whether you have a strong GPU. Since PPI training requires almost 8 GBs of VRAM
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    # device = torch.device("cuda" if torch.cuda.is_available() and not config['force_cpu'] else "cpu")
    config['num_heads_per_layer'] = [config['num_heads_per_layer1'], config['num_heads_per_layer2'], config['num_heads_per_layer3']]

    print({"num_heads_per_layer": config['num_heads_per_layer'],
                "num_of_layers": config['num_of_layers'],
                "num_features_per_layer": config['num_features_per_layer'],
                # "add_skip_connection": config['add_skip_connection'],
                # "bias": config['bias'],
                # "dropout": config['dropout']
                })
    # Step 1: prepare the data loaders
    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    # main_loop = get_main_loop(
    #     config,
    #     gat,
    #     loss_fn,
    #     optimizer,
    #     config['patience_period'],
    #     time.time())
    main_loop = MainLoop(
        config,
        gat,
        loss_fn,
        optimizer,
        config['patience_period'],
        time.time())

    # BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop.main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop.main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and micro-F1 on the test dataset. Friends don't let friends overfit to the test data. <3

    if config['should_test']:
        micro_f1 = main_loop.main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)
        config['test_perf'] = micro_f1

        print('*' * 50)
        print(f'Test micro-F1 = {micro_f1}')
    else:
        config['test_perf'] = -1

    session.report({"micro_f1": micro_f1})
    # Save the latest GAT in the binaries directory
    # torch.save(
    #     get_training_state(config, gat),
    #     os.path.join(BINARIES_PATH, get_available_binary_name(config['dataset_name']))
    # )


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=100)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)
    parser.add_argument("--force_cpu", type=bool, help='use CPU if your GPU is too small', default=False)

    # Dataset related (note: we need the dataset name for metadata and related stuff, and not for picking the dataset)
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.PPI.name)
    parser.add_argument("--batch_size", type=int, help='number of graphs in a batch', default=2)
    parser.add_argument("--should_visualize", type=bool, help='should visualize the dataset?', default=False)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=False)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=5)
    args = parser.parse_args('')

    # I'm leaving the hyperparam values as reported in the paper, but I experimented a bit and the comments suggest
    # how you can make GAT achieve an even higher micro-F1 or make it smaller
    gat_config = {
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [PPI_NUM_INPUT_FEATURES, 64, 64, PPI_NUM_CLASSES],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['ppi_load_test_only'] = False  # load both train/val/test data loaders (don't change it)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


train_gat_ppi(get_training_args())

print ('finished re-creating the paper result, start hyperparameter tuning')

from ray import air
import ray
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray import air
from ray.tune.stopper import (CombinedStopper,
MaximumIterationStopper, TrialPlateauStopper, ExperimentPlateauStopper)

ray.shutdown()
ray.init()

scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="micro_f1",
    mode="max",
    grace_period=1,
)
# tune_config=tune.TuneConfig(scheduler=scheduler),

stopper = CombinedStopper(
    MaximumIterationStopper(max_iter=10),
    TrialPlateauStopper(metric="micro_f1"),
)

tuner = tune.Tuner(
    tune.with_resources(train_gat_ppi, {"gpu": 1,'cpu': 12}),
    run_config=air.RunConfig(
      name="gnn_exp_heads_per_layer",
      stop=stopper,
      verbose=1,
  ),
    tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=1),
    param_space={
        # distribution for resampling
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        'num_of_epochs': 200,
         'patience_period': 50,
         'lr': 0.005,
         'weight_decay': 0,
         'should_test': True,
         'force_cpu': False,
         'dataset_name': 'PPI',
         'batch_size': 2,
         'should_visualize': False,
         'enable_tensorboard': False,
         'console_log_freq': 10,
         'checkpoint_freq': 5,
         'ppi_load_test_only': False,
        "num_of_layers": 3 , # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        'num_heads_per_layer1': tune.grid_search(list(range(3, 6))),
        "num_heads_per_layer2": tune.grid_search(list(range(3, 6))),  # other values may give even better results from the reported ones
        'num_heads_per_layer3': tune.grid_search(list(range(3, 6))),
        "num_features_per_layer": [PPI_NUM_INPUT_FEATURES, 64, 64, PPI_NUM_CLASSES],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.,  # dropout hurts the performance (best to keep it at 0)
        #         "lr": lambda: np.random.uniform(0.0001, 1),
        #         # allow perturbations within this set of categorical values
        #         "momentum": [0.8, 0.9, 0.99],
    }
)
results = tuner.fit()

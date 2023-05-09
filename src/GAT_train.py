import enum
import os
import torch
from utils import DEVICE, BINARIES_PATH, CHECKPOINTS_PATH, COMMIT_HASH_LOC
from sklearn.metrics import f1_score
from ray.air import session
import time
import git
import re  # regex



device=DEVICE

def get_training_state(training_config, model):
    training_state = {
        "commit_hash": git.Repo(COMMIT_HASH_LOC).head.object.hexsha,

        # Training details
        "dataset_name": training_config['dataset_name'],
        "num_of_epochs": training_config['num_of_epochs'],
        "test_perf": training_config['test_perf'],

        # Model structure
        "num_of_layers": training_config['num_of_layers'],
        "num_heads_per_layer": training_config['num_heads_per_layer'],
        "num_features_per_layer": training_config['num_features_per_layer'],
        "add_skip_connection": training_config['add_skip_connection'],
        "bias": training_config['bias'],
        "dropout": training_config['dropout'],

        # Model state
        "state_dict": model.state_dict()
    }

    return training_state


def print_model_metadata(training_state):
    header = f'\n{"*" * 5} Model training metadata: {"*" * 5}'
    print(header)

    for key, value in training_state.items():
        if key != 'state_dict':  # don't print state_dict it's a bunch of numbers...
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')


# This one makes sure we don't overwrite the valuable model binaries (feel free to ignore - not crucial to GAT method)
def get_available_binary_name(dataset_name='unknown'):
    prefix = f'gat_{dataset_name}'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'



class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2

class MainLoop:
    def __init__(self, config, gat, sigmoid_cross_entropy_loss, optimizer, patience_period, time_start):
        self.config = config
        self.gat = gat
        self.sigmoid_cross_entropy_loss = sigmoid_cross_entropy_loss
        self.optimizer = optimizer
        self.patience_period = patience_period
        self.time_start = time_start
        self.device = next(gat.parameters()).device

    def main_loop(self, phase, data_loader, epoch=0):
        global BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT

        config = self.config
        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            self.gat.train()
        else:
            self.gat.eval()

        # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
        # We merge them into a single graph with 2 connected components, that's the main idea. After that
        # the implementation #3 is agnostic to the fact that those are multiple and not a single graph!
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
            # Push the batch onto GPU - note PPI is to big to load the whole dataset into a normal GPU
            # it takes almost 8 GBs of VRAM to train it on a GPU
            edge_index = edge_index.to(device)
            node_features = node_features.to(device)
            gt_node_labels = gt_node_labels.to(device)

            # I pack data into tuples because GAT uses nn.Sequential which expects this format
            graph_data = (node_features, edge_index)

            # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
            # shape = (N, C) where N is the number of nodes in the batch and C is the number of classes (121 for PPI)
            # GAT imp #3 is agnostic to the fact that we actually have multiple graphs
            # (it sees a single graph with multiple connected components)
            nodes_unnormalized_scores = self.gat(graph_data)[0]

            # Example: because PPI has 121 labels let's make a simple toy example that will show how the loss works.
            # Let's say we have 3 labels instead and a single node's unnormalized (raw GAT output) scores are [-3, 0, 3]
            # What this loss will do is first it will apply a sigmoid and so we'll end up with: [0.048, 0.5, 0.95]
            # next it will apply a binary cross entropy across all of these and find the average, and that's it!
            # So if the true classes were [0, 0, 1] the loss would be (-log(1-0.048) + -log(1-0.5) + -log(0.95))/3.
            # You can see that the logarithm takes 2 forms depending on whether the true label is 0 or 1,
            # either -log(1-x) or -log(x) respectively. Easy-peasy. <3
            loss = self.sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

            if phase == LoopPhase.TRAIN:
                self.optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                self.optimizer.step()  # apply the gradients to weights

            # Calculate the main metric - micro F1, check out this link for what micro-F1 exactly is:
            # https://www.kaggle.com/enforcer007/what-is-micro-averaged-f1-score

            # Convert unnormalized scores into predictions. Explanation:
            # If the unnormalized score is bigger than 0 that means that sigmoid would have a value higher than 0.5
            # (by sigmoid's definition) and thus we have predicted 1 for that label otherwise we have predicted 0.
            pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            #
            # Logging
            #

            global_step = len(data_loader) * epoch + batch_idx
            if phase == LoopPhase.TRAIN:
                # Log metrics
                if config['enable_tensorboard']:
                    session.report({"training_loss": loss.item(), 'global_step': global_step})
                    session.report({"training_micro_f1": micro_f1, 'global_step': global_step})

                    # Log to console
                if config['console_log_freq'] is not None and epoch % config[
                    'console_log_freq'] == 0 and batch_idx == 0:
                    print(f'GAT training: time elapsed= {(time.time() - self.time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | train micro-F1={micro_f1}.')

                # Save model checkpoint
                if config['checkpoint_freq'] is not None and (epoch + 1) % config[
                    'checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                    config[
                        'test_perf'] = -1  # test perf not calculated yet, note: perf means main metric micro-F1 here
                    torch.save(get_training_state(config, self.gat),
                               os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

            elif phase == LoopPhase.VAL:
                # Log metrics
                if config['enable_tensorboard']:
                    session.report({"val_loss": loss.item(), 'global_step': global_step})
                    session.report({"val_micro_f1": micro_f1, 'global_step': global_step})

                    # Log to console
                if config['console_log_freq'] is not None and epoch % config[
                    'console_log_freq'] == 0 and batch_idx == 0:
                    print(f'GAT validation: time elapsed= {(time.time() - self.time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | val micro-F1={micro_f1}')

                # The "patience" logic - should we break out from the training loop? If either validation micro-F1
                # keeps going up or the val loss keeps going down we won't stop
                if micro_f1 > BEST_VAL_MICRO_F1 or loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_MICRO_F1 = max(micro_f1,
                                            BEST_VAL_MICRO_F1)  # keep track of the best validation micro_f1 so far
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best micro_f1
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= self.patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

            else:
                return micro_f1  # in the case of test phase we just report back the test micro_f1
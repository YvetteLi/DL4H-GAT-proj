# Reproducibility Project for CS598 DL4H in Spring 2023

## Original Paper
[Graph Attention Networks](https://arxiv.org/abs/1710.10903).
Special thanks to the [implementation](https://github.com/gordicaleksa/pytorch-GAT/tree/main) from Aleksa Gordić

### Session-based Social Network

- [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)
   - [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN/tree/master)
   - [SR-GNN-geometric](https://github.com/userbehavioranalysis/SR-GNN_PyTorch-Geometric)
- [Graph Contextualized Self-Attention Network for Session-based Recommendation](https://www.ijcai.org/proceedings/2019/0547.pdf)
   - [GC-SAN](https://github.com/johnny12150/GC-SAN/)

## Data Acquisition and Dependencies

### A. Install required packages

On MacOS or Linux machine, please do

```pip install -r requirements.txt```

On Google Collab, comment out pytorch depency and run 
```!apt-get install libcairo2-dev libjpeg-dev libgif-dev``` to install pycairo. 

```text
# requirements.txt
torch==2.0.0+cu118
numpy
igraph
pycairo
cairocffi
networkx
ray==2.4
tensorboardX
torch_geometric
GitPython
scikit-learn
```


### B. Datasets
The dataset needed to reproduce the experiments
- PPI: <https://data.dgl.ai/dataset/ppi.zip>
- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup> or <https://competitions.codalab.org/competitions/11161>


Before preprocessing the DIGINETICA data, the header needed to be added to the train-item-views.csv

```sed -i '1s/^/sessionId;userId;itemId;timeframe;eventdate/' train-item-views.csv```

## Preprocessing

Create PPI datasets by command `cd src; python dataset-GAT.py`

Create diginetica datasets by command `cd datasets; python preprocess.py --dataset=diginetica`

Other args include (referenced from original SR-GNN github)
```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```


## Training

To reproduce GAT and hyperparameter tuning results, run 

```cd src; python GAT_train.py```

<em>It is recommended to run the GAT model on Google Collab for best results</em>

To reproduce RS-GNN and its variant result, run

```bash
cd src; python main.py --dataset=diginetica --model=GNN # for SR-GNN model
cd src; python main.py --dataset=diginetica --model=GCSAN # for GC-SAN model
cd src; python main.py --dataset=diginetica --model=SRGAT # for SR-GAT model
```

Other args include (referenced from original SR-GNN github)

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size - 100 for GNN, 50 for GC-SAN
  --hiddenSize HIDDENSIZE
                        hidden state size - 100 for GNN, 120 for GC-SAN
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
...
```

## Results
### Grid Search Results in Number of Heads
| num_heads_per_layer1 | num_heads_per_layer2 | num_heads_per_layer3 | total time (s) | micro_f1 |
|----------------------|----------------------|----------------------|----------------|----------|
| 3                    | 4                    | 5                    | 329            | 0.852    |
| 3                    | 4                    | 3                    | 327            | 0.863    |
| 3                    | 3                    | 4                    | 248            | 0.870    |
| 4                    | 3                    | 4                    | 326            | 0.872    |
| 5                    | 4                    | 3                    | 329            | 0.906    |
| 5                    | 5                    | 5                    | 332            | 0.927    |
| 4                    | 4                    | 3                    | 305            | 0.927    |
| 5                    | 5                    | 3                    | 177            | 0.943    |
| 3                    | 3                    | 5                    | 329            | 0.945    |
| 3                    | 3                    | 3                    | 328            | 0.956    |
| 4                    | 3                    | 5                    | 328            | 0.965    |
| 3                    | 4                    | 4                    | 327            | 0.971    |
| 5                    | 3                    | 4                    | 326            | 0.971    |
| 4                    | 4                    | 5                    | 329            | 0.972    |
| 4                    | 3                    | 3                    | 327            | 0.973    |
| 5                    | 3                    | 3                    | 332            | 0.975    |
| 5                    | 3                    | 5                    | 328            | 0.976    |
| 4                    | 4                    | 4                    | 326            | 0.976    |
| 3                    | 5                    | 4                    | 328            | 0.977    |
| 3                    | 5                    | 5                    | 329            | 0.977    |
| 3                    | 5                    | 3                    | 331            | 0.978    |
| 4                    | 5                    | 3                    | 329            | 0.978    |
| 4                    | 5                    | 4                    | 331            | 0.978    |
| 4                    | 5                    | 5                    | 334            | 0.980    |
| 5                    | 4                    | 4                    | 330            | 0.982    |
| 5                    | 4                    | 5                    | 333            | 0.982    |
| 5                    | 5                    | 4                    | 329            | 0.984    |

### Learning Rate Exploration
| Experiment | learning rate | Time    | micro-f1 |
|------------|---------------|---------|----------|
| 1          | 0.00105       | 405.348 | 0.970019 |
| 2          | 0.00048       | 99.9874 | 0.787092 |
| 3          | 0.00019       | 15.5681 | 0.416091 |
| 4          | 0.00050       | 15.4964 | 0.398843 |
| 5          | 0.00017       | 15.921  | 0.399886 |
| 6          | 0.00017       | 15.75   | 0.400449 |
| 7          | 0.00041       | 15.7218 | 0.399741 |
| 8          | 0.00047       | 16.2647 | 0.406799 |
| 9          | 0.00025       | 15.3199 | 0.400362 |
| 10         | 0.00046       | 15.3425 | 0.404221 |

### Hidden Dimension Exploration
| num_features_layer1 | num_features_layer2 | total time (s) | micro_f1 |
|---------------------|---------------------|----------------|----------|
| 256                 | 64                  | 129.68         | 0.945    |
| 128                 | 256                 | 137.51         | 0.957    |
| 64                  | 64                  | 111.84         | 0.958    |
| 128                 | 64                  | 113.97         | 0.961    |
| 256                 | 128                 | 137.14         | 0.966    |
| 64                  | 128                 | 115.53         | 0.976    |
| 64                  | 256                 | 131.60         | 0.978    |
| 128                 | 128                 | 121.95         | 0.981    |
| 256                 | 256                 | 156.85         | 0.983    |


### Diginetica Results with RS-GNN, RS-GAT and GC-SAN
|  |SR-GNN  | SR-GNN | SR-GAT |SR-GAT  | GC-SAN | GC-SAN |
|---------|---------|--------|---------|--------|---------|--------|
| Epoch   | Prec@20 | MRR@20 | Prec@20 | MRR@21 | Prec@20 | MRR@22 |
| 1       | 50.71   | 17.85  | 50.15   | 17.41  | 48.28   | 16.74  |
| 2       | 50.38   | 17.93  | 50.17   | 17.42  | 48.27   | 16.69  |
| 3       | 50.53   | 17.85  | 49.92   | 17.36  | 47.94   | 16.54  |
| 4       | 50.93   | 17.93  | 50.28   | 17.10  | 48.30   | 16.33  |
| 5       | 50.38   | 17.92  | 49.92   | 17.33  | 48.11   | 16.69  |

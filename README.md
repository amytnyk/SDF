# Homework: Implicit SDF representation

## Usage


### Preparation

* Install all required modules from `requirements.txt`
* Prepare `models` directory with all .obj mesh files

### Training

```shell
python3 train_all.py --input models --output output
```

Detailed description:
```text
usage: train_all.py [-h] --input INPUT --output OUTPUT [--batch BATCH] [--epochs EPOCHS] [--lr LR]

options:
  -h, --help       show this help message and exit
  --input INPUT    Path to the input objects directory
  --output OUTPUT  Path to the output directory
  --batch BATCH    Batch size
  --epochs EPOCHS  Number of epochs
  --lr LR          Learning rate
```

### Evaluation

```shell
python3 evaluate_all.py --models output --meshes models/ --output results/results.csv
```

Detailed description:
```text
usage: evaluate_all.py [-h] --models MODELS --meshes MESHES --output OUTPUT [--points POINTS]

options:
  -h, --help       show this help message and exit
  --models MODELS  Path to the trained models
  --meshes MESHES  Path to the original objects
  --output OUTPUT  Output csv path
  --points POINTS  Number of points
```

### Mesh generating

```shell
python3 generate_mesh.py --model output/25/best.pth --output results/25.ply
```

Detailed description:
```text
usage: generate_mesh.py [-h] --model MODEL --output OUTPUT [--resolution RESOLUTION]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to the trained model
  --output OUTPUT       Path to the output .ply file
  --resolution RESOLUTION
                        Resolution
```

## Report

Implementation details and conclusion are here - [REPORT.md](REPORT.md)
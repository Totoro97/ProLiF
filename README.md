# ProLiF
We present a novel neural light field representation for efficient view synthesis.
![](./static/giants-1m.gif)

## [Project page](https://totoro97.github.io/projects/prolif/) |  [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Data](https://drive.google.com/drive/folders/xxxx)
This is the official repo for the implementation of **Progressively-connected light field network for efficient view synthes**.

## How to run?
### Setup

Clone this repository

```shell
git clone https://github.com/Totoro97/ProLiF.git
cd ProLiF
pip install -r requirements.txt
```

### Training

- **Training for novel view synthesis**

```shell
python exp_runner.py --config-name=prolif case_name=<case_name>
```

- **Training for scene fitting under varing light conditions**

```shell
python exp_runner.py --config-name=prolif-lpips case_name=<case_name>
```

- **Training for text-guided scene style editing** 

```shell
python exp_runner.py --config-name=prolif_clip case_name=<case_name>
```

### Testing

- **Rendering all test views** 
```shell
python exp_runner.py --config-name=<config_name> case_name=<case_name> mode=validate is_continue=true  # use latest checkpoint
```
The synthesized images can be found in `exp/<case_name>/<exp_name>/validation`.

- **Rendering video** 
```shell
python exp_runner.py --config-name=<config_name> case_name=<case_name> mode=video is_continue=true  # use latestcheck point
```
The synthesized video can be found in `exp/<case_name>/<exp_name>/video`.

### Train ProLiF with your custom data
We follow the same data convention as [LLFF](https://github.com/Fyusion/LLFF). You may follow the original LLFF instruction for data preparation.

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{wang2021prolif,
  author    = {Wang, Peng and Liu, Yuan and Lin, Guying and Gu, Jiatao and Liu, Lingjie and Komura, Taku and Wang, Wenping},
  title     = {Progressive-connected Light Field Network for Efficient View Synthesis},
  journal   = {Arxiv},
  year      = {2022},
}
```


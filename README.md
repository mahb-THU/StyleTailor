# StyleTailor

## Installation

``` bash
conda create -n styletailor python=3.10
cd /code
pip install -r requirements.txt
```

``` bash
conda create -n styletailor_eval python=3.10
pip install pyiqa
```

## Inference

``` bash
conda activate styletailor
python pipeline.py
```

```bash
conda activate styletailor_eval
cd /code/utils
python eval.py
```
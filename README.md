## Enhancing Source Code Summarization with a Hierarchical Structure of Program Dependency Graph-Induced Transformer

### Installing HiSPIT

Require `Linux` and `Python 3.6` or higher. It also requires installing `PyTorch` version 1.3 or higher. `CUDA` is strongly recommended for speed, but not necessary.

Its other dependencies are listed in `requirements.txt`.

Run the following commands to clone the repository and install HiSPIT:

```
git clone https://github.com/jianalex/HiSPIT.git
```

```
conda create --name HiSPIT python=3.8
conda activate HiSPIT
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

```
conda install numpy tqdm nltk prettytable torch>=1.3.0 psutil matplotlib seaborn
```

```
cd HiSPIT
python setup.py develop
```

### Training/Testing Models

```
$ cd  scripts/java
```


To train/evaluate a model, run:

```
$ bash script_name.sh GPU_ID MODEL_NAME
```

For example, to train/evaluate the transformer model, run:

```
$ bash transformer.sh 0 my_model
```

#### Generated log files

While training and evaluating the models, a list of files are generated inside a `tmp` directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).


**Structure of the JSON files** 
Each line in a JSON file is a JSON object. An example is provided below.

```json 
{
    "id": 0,
    "code": "private int current Depth ( ) { try { Integer one Based = ( ( Integer ) DEPTH FIELD . get ( this ) ) ; return one Based - NUM ; } catch ( Illegal Access Exception e ) { throw new Assertion Error ( e ) ; } }",
    "predictions": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "references": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "bleu": 1,
    "rouge_l": 1
}
```


#### Running experiments on CPU/GPU/Multi-GPU

- If GPU_ID is set to -1, CPU will be used.
- If GPU_ID is set to one specific number (e.g., 0), only one GPU will be used.
- If GPU_ID is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.

### Acknowledgement

We borrowed and modified code from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum), [SiT](https://github.com/gingasan/sit3). We would like to expresse our gratitdue for the authors of these repositeries.


### Citation

```

```


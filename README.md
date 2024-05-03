# Finding the best approach to classify music genre in GTZAN database 
> Author: Tianhao Liu (20205784)

> Email: liutianhao328@gmail.com | tianhao.liu@ucdconnect.ie

**In this project, I tried two approachs to classify the genres of songs. One is to build deeplearning model from scratch, using CNN, MLP, LSTM, and GRU. Beside this, I also tried to fine tune exsiting pre-trained models (vgg19, resNext, and SqueezeNet)**


## Train From Scratch
### Quick Start
1. Download the GTZAN dataset to this project folder
2. Unzip the downloaded database, the name of the downloaded database should be `Data`
3. If the database is in somewhere else or with other names, please write the path to config file `hparams.yaml`. You need to modify the `audio_dir` and `image_dir` in it. 
4.  `main.ipynb` is all you need. 
5.  After training, a trained model will be saved in folder `checkpoints`, and its loss record will be saved in folder   `logs`. 


| Model | Status |
|-------|--------|
| MLP   |   ✅   |
| CNN   |   ✅   |
| LSTM  |   ✅   |
| GRU   |   ✅   |


## Fine Tune
1. install all the packages declared in `requirements.txt`
2. I have fine tuned 3 pre-trained models, that are: `resnext`, `vgg19`, and `squeezenet`. You can find them in `finetune-resnext.ipynb`, `finetune-vgg.ipynb`, and `finetune-squeezenet.ipynb` respectively. 

| Model | Status |
|-------|--------|
| VGG    |   ✅   |
| ResNext   |   ✅   |
| SqueezeNet  |   ✅   |

## Techniques in Training

| Feature               | Status |
|-----------------------|--------|
| Early Stop            |   ✅   |
| Batch training        |   ✅   |
| Checkpoint            |   ✅   |
| Log (loss)            |   ✅   |
| Train-test-split      |   ✅   |
| Evaluation            |   ✅   |

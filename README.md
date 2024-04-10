## Comparing the Performance of Different Deep Learning Models on the GTZAN Dataset
> Author: Tianhao Liu 
> Email: liutianhao328@gmail.com | tianhao.liu@ucdconnect.ie

### Quick Start
1. Download the GTZAN dataset to this project folder
2. Unzip the downloaded database, the name of the downloaded database should be `Data`
3. If the database is in somewhere else or with other names, please write the path to config file `hparams.yaml`. You need to modify the `audio_dir` and `image_dir` in it. 
4.  `main.ipynb` is all you need. 
5.  After training, a trained model will be saved in folder `checkpoints`, and its loss record will be saved in folder   `logs`. 


### Add your own model
To easily add new models, I designed the training process to be a highly scalable. All you need to do is:
1. Add your own model in folder `models`
2. Create the instance of your model in `models/model_manager.py`
3. Add the model parameters in `hparams.yaml`
4. Run your model on GTZAN dataset `python train.py <your model name specified in hparams.yaml>`


### Models and Training:

| Model | Status |
|-------|--------|
| MLP   |   ✅   |
| CNN   |   ✅   |
| LSTM  |   ✅   |
| GRU   |   ✅   |


| Feature               | Status |
|-----------------------|--------|
| Early Stop            |   ✅   |
| Batch training        |   ✅   |
| Checkpoint            |   ✅   |
| Log (loss)            |   ✅   |
| Train-test-split      |   ✅   |
| Evaluation            |   ✅   |

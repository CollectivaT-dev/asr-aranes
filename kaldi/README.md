# Scripts for training a Kaldi speech recognition engine for Aranes with Common Voice dataset

Kaldi scripts to train a Kaldi-based ASR engine for Aranes using Common Voice corpus.

## Manual installation (Linux)

You need to first install Kaldi and SRILM. We provide the instructions, however you should make sure to follow the official guidelines in [Kaldi repository](https://github.com/kaldi-asr/kaldi):

```
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j 4
cd ../src
./configure --shared
make -j clean depend
make -j 8
cd ..
```

To install SRILM:

```
/opt/kaldi/tools/install_srilm.sh <name> <company> <email>
```

Then you should clone this repository under `egs` directory of `kaldi`:

```
cd egs
git clone https://github.com/CollectivaT-dev/asr-aranes.git
cd asr-aranes/kaldi
```

Finally, make sure you also have Python 3 and installed the required modules:

```
pip install tqdm pandas
```

## Docker installation

We provide a docker setup that takes care of all instalations. 

```
git clone https://github.com/CollectivaT-dev/asr-aranes.git
cd asr-aranes/kaldi/docker
docker build -t kaldidock kaldi
```

Once the image had been built, all you have to do is interactively attach to its bash terminal via the following command:

```
docker run -it -v <path-to-repo>:/opt/kaldi/egs/kaldi-aranes \
                 -v <path-to-corpus-base-directory>:/mnt \
                 --gpus all --name <container-name> <built-docker-name> bash
```

Once you finish this step, you should be in a docker container's bash terminal now to start the training.


## Training

All training scripts are inside `s5` directory: 

```
cd <kaldi-dir>/egs/kaldi-aranes/s5
```

If you're using GPU (and you should), make sure to flag them:

```
export CUDA_VISIBLE_DEVICES=0,1
```

To start training, all you need to do is call `run.sh` specifying a directory where to download the corpora: 

```
bash run.sh --corpusbase <corpus-base-directory>  #if running from docker <corpus-base-directory> is "/mnt"
``` 

To train toy models to see if all the process works smoothly, you can use the `subset` option. This will prepare a training dataset using only a specified number of samples:

```
bash run.sh --corpusbase <corpus-base-directory> --subset 1000
```

## Results

Evaluations are done on separately on testing portions of the two corpora. `run.sh` will print out WER scores at the end. 

```
To be published soon...
```


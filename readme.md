# install tensorflow with anaconda

Anaconda 5.0.1 (Python2.7 Version) download from[https://www.anaconda.com/download/#linux]


tensorflow official[https://www.tensorflow.org/install/install_linux#InstallingAnaconda]

## python 2.7
Python 2.7/CPU only:https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl

## python 3.5

>$ conda create -n tf14 python=3.5
>$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl
>$ source activate tf14


# check

$python
>>>import tensorflow

# use git samples

$ git clone https://github.com/tensorflow/models.git

* MNIST: models/official/mnist/
* Object Detect: models/research/object_detection/g3doc/



# retrain
* https://www.tensorflow.org/tutorials/image_retraining
* https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0


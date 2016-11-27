# TensorFlowを使ったDeepLearning学習

## 1.構築 Install Tensorflow

### 参考サイト
* https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation

### 1.1 dockerを使った環境構築

* dockerのインストール(省略)
* tensorflow docker image の取得
	* $ sudo docker pull b.gcr.io/tensorflow/tensorflow
		* Using default tag: latest
		* latest: Pulling from tensorflow/tensorflow
		* Status: Downloaded newer image for b.gcr.io/tensorflow/tensorflow:latest
* containerの作成
	* 公式サイトの通りのコマンドだと/run_jupyter.shが起動してしまうのでbashを指定する
		* $ sudo docker run -it b.gcr.io/tensorflow/tensorflow bash
		* /notebooks# cd
		* ~#

* Test the TensorFlow installation
	* ~# python
	* >>> import tensorflow as tf
	* >>> hello = tf.constant('Hello, TensorFlow!')
	* >>> sess = tf.Session()
	* >>> print(sess.run(hello))
	* Hello, TensorFlow!

* Run a TensorFlow demo model
	* check the path of tensorflow packages
		* ~# python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
		* /usr/local/lib/python2.7/dist-packages/tensorflow
		* ~# python -m tensorflow.models.image.mnist.convolutional
		* Extracting data/train-images-idx3-ubyte.gz
		* Extracting data/train-labels-idx1-ubyte.gz
		* Extracting data/t10k-images-idx3-ubyte.gz
		* Extracting data/t10k-labels-idx1-ubyte.gz
		* Initialized!
		* Step 0 (epoch 0.00), 12.6 ms
		* Minibatch loss: 12.054, learning rate: 0.010000
		* Minibatch error: 90.6%
		* Validation error: 84.6%
		* Step 100 (epoch 0.12), 346.4 ms
		* ...

### 1.2 dockerにgitをインストール
* install git
	* ~# sudo apt-get update
	* ~# sudo apt-get install git

### 1.3 docerをcommit（imageとして保存）
* commit container
	* Ctrl + P + Q
	* sudo docker ps -a
	* sudo docker commit <container name> <new image name>

### 1.4 ホストOSのディレクトリをDockerコンテナ内で利用する
* 参考:https://thinkit.co.jp/story/2015/09/15/6384
* sudo docker run -v /home/suzuki/docker/shared:/root/shared -it b.gcr.io/tensorflow/tensorflow bash

### 1.5 check the version of tensorflow ,then update
* http://scriptlife.hacca.jp/contents/programming/2016/08/14/post-1709/
* checking the version
	* ~# pip list
	* tensorflow (0.7.1)
* update
	* First update python because of SSL error when update tensorflow (to 0.11)
	* ~# apt-get update
	* ~# apt-get upgrade python
	* ~# pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl


## 2. Use TensorFlow function

### 参考サイト
* https://www.tensorflow.org/versions/r0.7/get_started/basic_usage.html#basic-usage
* https://www.tensorflow.org/versions/master/how_tos/index.html

### メモ

* Networkをgraph、Neuronをops(operations),input/outputデータをtensorsと呼ぶらしい。
* construction phaseでgraphを構築して、execution phaseで'session'を用いてgraph計算を行う。



## 3.サンプル実行

### 3.1. Retrain Inception with own images
* https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html#how-to-retrain-inceptions-final-layer-for-new-categories
* install bazel
	* install java
		* ~# sudo apt-get install software-properties-common python-software-properties
		* ~# sudo add-apt-repository ppa:webupd8team/java
		* ~# sudo apt-get update
		* ~# sudo apt-get install oracle-java8-installer
	* install bazel
		* ~# wget https://github.com/bazelbuild/bazel/releases/download/0.3.2/bazel-0.3.2-installer-linux-x86_64.sh
		* ~# chmod +x bazel-0.3.2-installer-linux-x86_64.sh
		* ~# ./bazel-0.3.2-installer-linux-x86_64.sh --user
* build tensorflow
	* ~# cd tensorflow
	* ~# ./configure
* install git
	* ~# sudo apt-get install git
* build train tool
	* ~# bazel build tensorflow/examples/image_retraining:retrain

* Trouble shoot
	* Error when build train tool
		* ERROR: /.../BUILD:1826:1: Linking of rule '//tensorflow/python:_pywrap_tensorflow.so' failed: gcc failed: error executing command ...
	* tensorflow 0.11 -> 0.12
		* git pull
	* rebuild
		* ~# ./coufigure

* train alter dataset
	* ~# bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos
	* -> /tmp/output_graph.pb output_labels.txt

### 3.2. for Raspberry Pi
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/pi_examples
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile

	* ~# tensorflow/contrib/makefile/download_dependencies.sh
	* ~# sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8
	* ~# cd tensorflow/contrib/makefile/downloads/protobuf/
	* ~# ./autogen.sh
	* ~# ./configure
	* ~# make
	* ~# sudo make install
	* ~# sudo ldconfig
	* ~# cd ../../../../..
	* ~# make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os" CXX=g++-4.8
		* it takes minuts
	* ~# sudo apt-get install -y libjpeg-dev

	* put trained model(tensorflow_inception_stripped.pb) and label(imagenet_comp_graph_label_strings.txt) in 'label_image/data' directory
	* ~# make -f tensorflow/contrib/pi_examples/label_image/Makefile

### 3.3. Object Detection
* https://github.com/Russell91/TensorBox



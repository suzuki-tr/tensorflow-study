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


## 2. Use TensorFlow function

### 参考サイト
* https://www.tensorflow.org/versions/r0.7/get_started/basic_usage.html#basic-usage
* https://www.tensorflow.org/versions/master/how_tos/index.html

### メモ

* Networkをgraph、Neuronをops(operations),input/outputデータをtensorsと呼ぶらしい。
* construction phaseでgraphを構築して、execution phaseで'session'を用いてgraph計算を行う。



## 3.サンプル実行

### 参考サイト
* https://github.com/Russell91/TensorBox



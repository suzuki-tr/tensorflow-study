# try TensorFlow for mobile

* https://www.tensorflow.org/mobile.html
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile

* デモやってみた
 * http://tensorflow.hatenablog.com/entry/2016/02/13/210000
 * https://blog.guildworks.jp/2015/11/16/tensorflow/

* Docker
 * https://hub.docker.com/r/ornew/tensorflow-android/

## 手順

* JDKインストール

 * $ sudo add-apt-repository ppa:webupd8team/java
 * $ sudo apt-get update
 * $ sudo apt-get install oracle-java8-installer

add-apt-repositoryがないと言われたら
 * $ sudo apt-get install software-properties-common python-software-properties


* Bazelインストール(build tool)

 * https://github.com/bazelbuild/bazel/releases
 * 「bazel-0.3.2-installer-linux-x86_64.sh」をダウンロード
 * $ wget https://github.com/bazelbuild/bazel/releases/download/0.3.2/bazel-0.3.2-installer-linux-x86_64.sh
 * $ chmod +x bazel-0.3.2-installer-linux-x86_64.sh
 * $ ./bazel-0.3.2-installer-linux-x86_64.sh --user

* 環境変数(AndroidSDK,NDKのパスを追加)

 * gedit ~/.bashrc
 * export ANDROID_HOME="$HOME/bin/android-sdk-linux"
 * export NDK_HOME="$HOME/bin/android-ndk-r13"
 * export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools:$NDK_HOME:$HOME/bin
 * source ~/.bashrc

* Android NDK

 * https://developer.android.com/ndk/downloads/index.html
 * 「android-ndk-r13-linux-x86_64.bin」をダウンロード
 * $ wget https://dl.google.com/android/repository/android-ndk-r13-linux-x86_64.zip
 * $ unzip android-ndk-r13-linux-x86_64.zip
# * $ chmod +x android-ndk-r13-linux-x86_64.bin
# * $ ./android-ndk-r13-linux-x86_64.bin

* Android SDK

 * https://developer.android.com/sdk/index.html
 * $ wget https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz
 * $ tar zxvf android-sdk_r24.4.1-linux.tgz
 * $ android
 * Android SDK ManagerにてSDKパッケージ追加（Android SDK Build-tools(Rev.24.0.3)、Android6.0(API23)、Android7.0(API24)）

  * Android Studio使ってもビルドできる？

* TensorFlow

 * $ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
 * $ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O ~/inception5h.zip
 * $ cd tensorflow
 * $ unzip ~/inception5h.zip -d tensorflow/examples/android/assets/
 * $ gedit WORKSPACE
	* uncomment lines of Android SDK/NDK
	* modify path
* Build

 * $ bazel build //tensorflow/examples/android:tensorflow_demo
 #* $ bazel build //tensorflow/examples/android:tensorflow_demo -c opt --copt=-mfpu=neon


 * $ adb install -r -g bazel-bin/tensorflow/examples/android/tensorflow_demo.apk

* Trouble
 * SDK api_level=24, NDK api_level=21  -> Install failed (Failure [INSTALL_FAILED_OLDER_SDK] Target Device is Android 4.4)
 * SDK api_level=19, NDK api_level=21  -> Error: /tmp/android_resources_tmp8267779757322785907/merged_resources/values/values.xml:29: error: Error retrieving parent for item: No resource found that matches the given name 'android:Theme.Material.Light.NoActionBar.Fullscreen'.
 * SDK api_level=19, NDK api_level=19

 * AndroidManifest.xml minSdkVersion=19,targetSdkVersion=19, WORKSPACE SDK api_level=24, NDK api_level=21
   -> Install OK , execution error

 * AndroidManifest.xml minSdkVersion=19,targetSdkVersion=19, WORKSPACE SDK api_level=19, NDK api_level=21
   -> Build Error "No resource found that matches the given name 'android:Theme.Material.Light'"

 * AndroidManifest.xml minSdkVersion=22,targetSdkVersion=22, WORKSPACE SDK api_level=22, NDK api_level=21
   -> tensorflow/examples/android/src/org/tensorflow/demo/CameraActivity.java:50: error: method does not override or implement a method from a supertype

 * AndroidManifest.xml minSdkVersion=22,targetSdkVersion=22, WORKSPACE SDK api_level=23, NDK api_level=21




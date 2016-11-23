# Docker: how to use

### 参考サイト

* http://docker.yuichi.com/index.html
* http://www.atmarkit.co.jp/ait/articles/1405/16/news032.html
* https://docs.docker.com/engine/installation/linux/ubuntulinux/
* http://tracpath.com/works/devops/how_to_install_the_docker/
* http://www.slideshare.net/zembutsu/introduction-to-docker-management-and-operations-basic

### 構築

* $ sudo apt-get update
(* $ sudo apt-get install docker-engine)
* $ sudo apt-get install docker.io

### 操作

* $ sudo docker info                                    :情報表示
* $ sudo docker version                                 :Client/Server状態
* $ sudo docker pull ubuntu:latest                      :Ubuntu Docker Image download
* $ sudo docker images                                  :List up Docker images
* $ sudo docker ps -a                                   :List up Docker container
* $ sudo docker run -it --name ubuntu1 ubuntu /bin/bash :create new container from image and run bash
	* docker run <option> <image> <command>
	* -i means 標準入力
	* -t means 擬似ターミナル
	* -d means detach起動(daemon用)
* $ sudo docker run -v /home/user/shared:/root/shared -it --name ubuntu1 ubuntu /bin/bash :create new container from image and run bash with shared directory
* commit container					:commit to image
	* Ctrl + P + Q
	* sudo docker ps -a
	* sudo docker commit <container name> <new image name>
* $ sudo docker stop
* $ sudo docker start -i ubuntu1                        :container restart
* $ sudo docker rm <container id>			:remove container
* $ sudo docker rmi <image name>			:remove image
* ~# exit						:コンテナの停止

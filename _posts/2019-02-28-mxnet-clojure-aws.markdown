---
title:  "Getting started with Clojure and MXNet on AWS"
layout: post
date: 2019-02-28 00:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- aws
- deeplearning
star: true
category: blog
author: arthurcaillau
description: Learn how to get started with MXNet and Clojure on AWS
---

Setting up a Deep Learning development box is often a tedious task that requires one to install the proper drivers and toolkits for the given Deep Learning Framework.

This post will cover how to setup such a box for MXNet and Clojure so that any Clojurist can start playing with MXNet on GPUs.

## MXNet

MXNet is the Deep Learning Framework developped jointly by Amazon and Microsoft. It is an incubating Apache Project.

* [MXNet Homepage][1]
* [MXNet project repository][2]

## MXNet and Clojure

Long story short: by leveraging Scala, Clojure can talk to the MXNet bindings. And this is amazing! Here is the [Clojure Package][3] in the MXNet repository.

## AWS Deep Learning AMI

AWS provides AMIs tailored to Deep Learning frameworks. They are pre-configured environments to quickly build deep learning applications.
The AMI we will cover in this post is the **AWS Deep Learning AMI (base) image** - [Find it here][4].

## Installation steps

Below are the steps one needs to follow to start playing with MXNet and Clojure on GPUs

1. Launch an EC2 instance
2. Connect to the EC2 instance
3. Update Java to Java 8
4. Update Cuda version
5. Install leiningen
6. Play with Clojure MXNet Examples

#### Launch an EC2 instance

Launch an EC2 instance with the AWS Deep Learning AMI base image. In this post I assume you select the **Deep Learning Base (Amazon Linux)** AMI.

![AMI Selection](/assets/images/aws-dl-ami/aws-ec2-ami-deep-learning-base.png)

Then pick a GPU instance type. `p2.xlarge` shoud be enough to get you started and will cost you around `$0.90` per hour. **Do not forget to terminate it after you are done with your Neural Network training**.

![Instance Type Selection](/assets/images/aws-dl-ami/aws-ec2-ami-instance-type.png)

#### Connect to the EC2 instance

Use the `pemfile` or username/password pair that was created to connect to your EC2 instance
```
ssh -i "pemfilename.pem" ec2-user@ec2-1-42-42-0.compute-1.amazonaws.com
```

#### Update Java to Java 8

From the EC2 instance, run the following command and select `java 8` that comes with the AMI
```
sudo update-alternatives --config java
```

#### Update Cuda version

From the EC2 instance, run the following commands to use the proper Cuda version required to run MXNet. For `MXNet 1.4.0`, one needs `cuda-9.2`

```
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-9.2 /usr/local/cuda
```

#### Install leiningen

Run the following commands to install the latest `leiningen`

```
wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein
chmod a+x lein
mv lein /usr/bin
lein
```

#### Play with Clojure MXNet Examples

First, one needs to clone the MXNet codebase that comes with Clojure examples
```
git clone https://github.com/apache/incubator-mxnet.git
```
Navigate the filesystem to get to the Clojure examples
```
cd incubator-mxnet/contrib/clojure-package/examples
```
We will run the code from the `module` example
```
cd module
```
Edit `project.clj` file to use the `gpu` library
```clojure
(defproject module-examples "0.1.0-SNAPSHOT"
  :description "Clojure examples for module"
  :plugins [[lein-cljfmt "0.5.7"]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 ;; This line below is important
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu "1.4.0"]]
  :pedantic? :skip
  :repositories
  [["staging" {:url "https://repository.apache.org/content/repositories/staging"
               :snapshots true
               :update :always}]
   ["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"
                 :snapshots true
                 :update :always}]]
  :main mnist-mlp)
```
One can monitor the GPU processes with the following command
```
watch nvidia-sim
```

![nvidia-sim output](/assets/images/aws-dl-ami/aws-ec2-ami-nvidia-sim.png)

Now it is time to run some Clojure code and get these Neural Networks to learn something!

Run the code from the REPL
```
lein repl
mnist-mlp=> (run-all [(context/gpu)])
```

Or run the code from a leiningen command
```
lein run :gpu
```

## Conclusion

Congratulations! If you have followed along, you are now able to harness the power of MXNet with Clojure on GPUs! Again, **do not forget to terminate your instance once you are done**.

Next time, we will cover what MXNet let us do and we will learn how to train Deep Learning models.

## References and Resources

* [MXNet Website][1]
* [MXNet Github Repo][2]
* [MXNet Clojure Package Github][3]
* [AWS Deep Learning AMIs][4]

[1]: https://mxnet.incubator.apache.org
[2]: https://github.com/apache/incubator-mxnet/
[3]: https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package
[4]: https://aws.amazon.com/machine-learning/amis/

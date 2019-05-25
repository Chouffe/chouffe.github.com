---
title:  "Running Clojure MXNet on Jupyter Notebook"
layout: post
date: 2019-05-22 00:00
image: /assets/images/jupyter-logo.jpeg
headerImage: true
published: false
tag:
- clojure
- mxnet
- jupyter
- notebook
- aws
- deeplearning
star: true
category: blog
author: arthurcaillau
description: Learn how to setup Jupyter Notebooks with Clojure MXNet locally and on AWS
---

Data Science work is often done with [Jupyter][1] Notebooks. Notebooks allow you to create and share documents that contain live code, equations, visualization and narrative text.

This post will explain how to setup Jupyter Notebooks to run MXNet locally and on a GPU with an EC2 instance.

## Lein Jupyter

### Installation

There is a [leiningen plugin][3] available for running Jupyter Notebooks with a Clojure Kernel.

First, create a new leiningen project
```bash
lein new jupyter-playground
```

Then open `project.clj` and add the `lein jupyter` plugin
```clojure
(defproject jupyter-playground "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  ;; Add the lein-jupyter plugin
  :plugins [[lein-jupyter "0.1.16"]]
  :dependencies [[org.clojure/clojure "1.10.0"]])
```

One needs to install the clojure kernel with the following command when running it for the first time

```bash
lein jupyter install-kernel
```

Now one can start `jupyter notebook` with the following command
```bash
lein jupyter notebook
```

### Evaluate Clojure Code with Jupyter

Click on the top right corner to create a new notebook. Select the `Lein-Clojure` kernel.

![Creating a new Notebook](/assets/images/lein-jupyter-tutorial-new.png){: .center-image }

One can now evaluate some clojure code inside a code cell

![Evaluating Clojure Code](/assets/images/jupyter-eval-clojure.png)


## Running MXNet with in a Jupyter Notebook

### Installing MXNet locally

First, clone the MXNet codebase that comes with examples and tutorials

```bash
git clone https://github.com/apache/incubator-mxnet.git
```

Navigate into the clojure root directory

```bash
cd contrib/clojure-package
```

Open `project.clj` to uncomment the package that matches your OS and hardware setup and comment out the CI package

```clojure
(defproject
  ...
  ;; Uncommenting this line: I am running on Linux and do not have a GPU
  [org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu "1.5.0-SNAPSHOT"]
  ;; Comment this line used for CI
  ; [org.apache.mxnet/mxnet-full_2.11 "INTERNAL"]
  ...
)
```

You should now be able to test that the clojure package can run locally

```bash
lein test
```

Install the `1.5.0-SNAPSHOT` jar locally by running

```bash
lein install
```

### Running MXNet code

The `examples` folder contains some Jupyter Notebook that one can run. We will be running the `BERT` Notebooks.

```bash
cd examples/bert
```

Start the Jupyter Notebook

```bash
lein jupyter notebook
```

Open the `fine-tune-bert.ipynb` notebook and play with the code

![Open Notebook](/assets/images/fine-tune-bert-notebook.png)

### Take Aways

One can now run arbitrary Clojure code from a Jupyter Notebook. Jupyter Notebooks are different from the Clojure REPL. I would argue that they are a great fit for Data Science because they allow to mix freely code, markdown and equations and can be shared and exported.

Running MXNet locally can be quite slow when we need to train large models. A GPU would be a better fit for that task. We will cover how we can make the Jupyter Notebook work on an EC2 instance with a GPU.

## On AWS EC2

Training Neural Networks requires a GPU to perform matrix operations faster. As an illustration, We will use a GPU instance from AWS to finetune the BERT model.
First, follow this [tutorial][2] to setup an EC2 instance with GPU that will be able to run MXNet.

Open an SSH tunnel on port 8888 with the EC2 instance you just created
```bash
ssh -i <file-key.pem> -L localhost:8888:localhost:8888 ubuntu@<instance-ip>
```

Navigate to the `bert` folder
```
cd contrib/clojure-package/examples/bert
```

Install the jar locally
```
lein install
```

Install the jupyter kernel
```
lein jupyter install-kernel
```

Start the Jupyter Notebook
```
lein jupyter notebook
```

One should now be able to open the URL displayed by the command above in your favorite browser `http://localhost:8888/?token=<token>`.

Open the `fine-tune-bert.ipynb` Notebook and follow the instructions for downloading the BERT model.
```
# Run from the Command Line the script to download BERT
./get_bert_data.sh
```

Evaluate all the code cells until the _Fine-tune BERT Model_ section. We need to set the device context to GPU like so

![Device Context GPU](/assets/images/bert-device-context-gpu.png)

Now you can run the last code cell that will fine tune BERT on the GPU. It should be very fast compared to running it on a CPU.
One can also check how the GPU is being used by running the command

```bash
watch nvidia-smi
```

![GPU usage](/assets/images/bert-gpu-usage.png)


## Conclusion

As clojurists, we can also leverage Jupyter Notebooks for our Data Science work. This post should give you all the necessary tooling to run arbitrary Clojure code on a GPU with a Jupyter Notebook.

I am looking forward to seeing all the cool MXNet models the community will train.

## References and Resources

* [Jupyter Website][1]
* [Getting started with Clojure and MXNet on AWS ][2]
* [Lein Jupyter plugin][3]
* [MXNet Clojure Package Github][4]

You can find the code used in this post in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials).

[1]: https://jupyter.org/
[2]: /mxnet-clojure-aws
[3]: https://github.com/clojupyter/lein-jupyter
[4]: https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package

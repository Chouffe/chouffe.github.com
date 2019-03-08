---
title:  "MXNet made simple: Clojure Symbol Visualization API"
layout: post
date: 2019-03-08 00:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- symbol
star: true
category: blog
author: arthurcaillau
description: Visualization of pretrained and user defined models with MXNet and Clojure
---

In this post we will look at the [MXNet visualization API][3]. We will learn how to visualize pretrained models and user defined models.

### Before we begin...

We will need to import certain packages:

```clojure
(require '[org.apache.clojure-mxnet.module :as m])
(require '[org.apache.clojure-mxnet.visualization :as viz])
```

### Pretrained Models

The [MXNet Model Zoo][1] is a central place for downloading state of the art pretrained models. One can download the model computation graphs and their trained parameters. It makes it straightforward to get started with making new predictions in no time.

We are going to download **VGG16** and **ResNet18**: two common state of the art models to perform computer vision tasks such as classification, segmentation, etc.

Below is the bash script for downloading **VGG16**.

```bash
#!/bin/bash

set -evx

mkdir -p model
cd model
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params
cd ..
```

* `vgg16-symbol.json`: computation graph of the **VGG16** model
* `vgg16-0000.params`: trained parameters and weights for the **VGG16** model

```bash
# Execute the bash script
$ chmod a+x download_vgg16.sh
$ sh download_vgg16.sh
```

And below is the bash script to download **ResNet18**

```bash
#!/bin/bash

set -evx

mkdir -p model
cd model
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params
cd ..
```

* `resnet-18-symbol.json`: computation graph of the **ResNet18** model
* `resnet-18-0000.params`: trained parameters and weights for the **ResNet18** model

```bash
# Execute the bash script
$ chmod a+x download_resnet18.sh
$ sh download_resnet18.sh
```

Make sure that the models are properly downloaded

```bash
$ cd model
$ ls
resnet-18-0000.params  resnet-18-symbol.json
vgg16-0000.params      vgg16-symbol.json
```

One can load the computation graph of a model using the **Module API**

```clojure
(def model-dir "model")

(def vgg16-mod
  "VGG16 Module"
  (m/load-checkpoint {:prefix (str model-dir "/vgg16") :epoch 0}))

(def resnet18-mod
  "Resnet18 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0}))
```

The visualization API uses [graphviz][2] under the hood to render computation graphs. We can write a small function that takes in the symbol to render and the path where to save the generated graphviz. By default, it generates pdf files as output format.

```clojure
(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
              model-sym
              {"data" input-data-shape}
              {:title model-name
               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))
```

Now we can visualize the pretrained models by calling this function

```clojure
(def model-render-dir "model_render")

;; Rendering pretrained VGG16
(render-model! {:model-name "vgg16"
                :model-sym (m/symbol vgg16-mod)
                :input-data-shape [1 3 244 244]
                :path model-render-dir})

;; Rendering pretrained Resnet18
(render-model! {:model-name "resnet18"
                :model-sym (m/symbol resnet18-mod)
                :input-data-shape [1 3 244 244]
                :path model-render-dir})
```

### User Defined Model

We can also visualize our own models with the same approach. We will define the **LeNet** model and visualize it with the Symbol Visualization API.

```clojure
(require '[org.apache.clojure-mxnet.symbol :as sym])

(defn get-symbol
  "Return LeNet Symbol

  Input data shape [`batch-size` `channels` 28 28]
  Output data shape [`batch-size 10]"
  []
  (as-> (sym/variable "data") data

    ;; First `convolution` layer
    (sym/convolution "conv1" {:data data :kernel [5 5] :num-filter 20})
    (sym/activation "tanh1" {:data data :act-type "tanh"})
    (sym/pooling "pool1" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Second `convolution` layer
    (sym/convolution "conv2" {:data data :kernel [5 5] :num-filter 50})
    (sym/activation "tanh2" {:data data :act-type "tanh"})
    (sym/pooling "pool2" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Flattening before the Fully Connected Layers
    (sym/flatten "flatten" {:data data})

    ;; First `fully-connected` layer
    (sym/fully-connected "fc1" {:data data :num-hidden 500})
    (sym/activation "tanh3" {:data data :act-type "tanh"})

    ;; Second `fully-connected` layer
    (sym/fully-connected "fc2" {:data data :num-hidden 10})

    ;; Softmax Loss
    (sym/softmax-output "softmax" {:data data})))
```

Now we can render it the same way as the pretrained models

```clojure
;; Rendering user defined LeNet
(render-model! {:model-name "lenet"
                :model-sym (get-symbol)
                :input-data-shape [1 3 28 28]
                :path model-render-dir})
```

### Rendered Models: VGG16, ResNet18 and LeNet

Here is a summary of the models we rendered in this tutorial

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width:100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-0pky{border-color:#ccc;border-width:1px;text-align:middle;vertical-align:top;font-family:bold; width:33%;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-0pky">VGG16</th>
    <th class="tg-0pky">ResNet18</th>
    <th class="tg-0pky">LeNet</th>
  </tr>
  <tr>
    <td class="tg-0lax">

      <a href="/assets/images/viz/vgg16.png" >
        <img class="small-image" src="/assets/images/viz/vgg16.png" alt="VGG16 Topology" />
      </a>

    </td>
    <td class="tg-0lax">

      <a href="/assets/images/viz/resnet18.png" >
        <img class="small-image" src="/assets/images/viz/resnet18.png" alt="ResNet18 Topology" />
      </a>
    </td>
    <td class="tg-0lax">

      <a href="/assets/images/viz/lenet.png" >
        <img class="small-image" src="/assets/images/viz/lenet.png" alt="LeNet Topology" />
      </a>
    </td>
  </tr>
</table>

## Conclusion

The Symbol VisualizationModule API** makes it simple to visualize any models: pretrained and user defined. It is good practice to make sure the topology of a model makes sense before training it or making predictions.

## References and Resources

* [MXNet Model Zoo][1]
* [Graphviz Website][2]
* [MXNet Visualization API Reference][3]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.lenet
  (:require [org.apache.clojure-mxnet.symbol :as sym]))

(defn get-symbol
  "Return LeNet Symbol

  Input data shape [`batch-size` `channels` 28 28]
  Output data shape [`batch-size 10]"
  []
  (as-> (sym/variable "data") data

    ;; First `convolution` layer
    (sym/convolution "conv1" {:data data :kernel [5 5] :num-filter 20})
    (sym/activation "tanh1" {:data data :act-type "tanh"})
    (sym/pooling "pool1" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Second `convolution` layer
    (sym/convolution "conv2" {:data data :kernel [5 5] :num-filter 50})
    (sym/activation "tanh2" {:data data :act-type "tanh"})
    (sym/pooling "pool2" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Flattening before the Fully Connected Layers
    (sym/flatten "flatten" {:data data})

    ;; First `fully-connected` layer
    (sym/fully-connected "fc1" {:data data :num-hidden 500})
    (sym/activation "tanh3" {:data data :act-type "tanh"})

    ;; Second `fully-connected` layer
    (sym/fully-connected "fc2" {:data data :num-hidden 10})

    ;; Softmax Loss
    (sym/softmax-output "softmax" {:data data})))
```

```clojure
(ns mxnet-clj-tutorials.visualization
  "Functions and utils to render pretrained and user defined models."
  (:require
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.visualization :as viz]

    [mxnet-clj-tutorials.lenet :as lenet]))

;; Run the `download_vgg16.sh` and `download_resnet18.sh`
;; prior to running the following code

(def model-dir "model")
(def model-render-dir "model_render")

;; Loading pretrained models

(def vgg16-mod
  "VGG16 Module"
  (m/load-checkpoint {:prefix (str model-dir "/vgg16") :epoch 0}))

(def resnet18-mod
  "Resnet18 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0}))

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
              model-sym
              {"data" input-data-shape}
              {:title model-name
               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))

(comment
  ;; Run the following function calls to render the models in `model-render-dir`

  ;; Rendering pretrained VGG16
  (render-model! {:model-name "vgg16"
                  :model-sym (m/symbol vgg16-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering pretrained Resnet18
  (render-model! {:model-name "resnet18"
                  :model-sym (m/symbol resnet18-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering user defined LeNet
  (render-model! {:model-name "lenet"
                  :model-sym (lenet/get-symbol)
                  :input-data-shape [1 3 28 28]
                  :path model-render-dir}))
```

[1]: https://mxnet.incubator.apache.org/model_zoo/
[2]: https://graphviz.org/
[3]: https://mxnet.incubator.apache.org/versions/master/api/clojure/docs/org.apache.clojure-mxnet.visualization.html

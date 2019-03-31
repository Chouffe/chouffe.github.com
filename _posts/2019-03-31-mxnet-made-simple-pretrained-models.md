---
title:  "MXNet made simple: Pretrained Models for image classification - Inception and VGG"
layout: post
date: 2019-03-31 00:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- inception
- vgg
- image classification
- model zoo
star: true
category: blog
author: arthurcaillau
description: Pretrained models with Clojure and MXNet for image classification
---

In this post, we will learn how to leverage pretrained models to perform image classification. The Computer Vision task is to associate a label with an unseen image. State of the art Machine Learning models have hundred of layers and require days and sometimes even weeks to train on GPUs. MXNet enables us to leverage such models very easily. We will see how to use three common Deep Learning models: **Inception and VGG**.

### Before we begin...

We will need to import certain packages:

```clojure
(require '[clojure.string :as string])

;; MXNet namespaces
(require '[org.apache.clojure-mxnet.module :as m])
(require '[org.apache.clojure-mxnet.ndarray :as ndarray])

;; OpenCV namespaces
(require '[opencv4.core :as cv])
(require '[opencv4.utils :as cvu])
```

## The MXNet Model Zoo

The [MXNet Model Zoo][5] is a set of pretrained models including the **computation graphs** and their **trained parameters**.
We can download the models from the Model Zoo. First, let's download **VGG16** by running the following bash script

```bash
#!/bin/bash

set -evx

mkdir -p model
cd model
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params
cd ..
```

* `vgg16-symbol.json`: serialized computation graph that can be loaded by MXNet
* `vgg16-0000.params`: binary file containing the `NDArrays` of the trained parameters

Let's also download the **Inception** pretrained models

```bash
#!/bin/bash

set -evx

mkdir -p model
cd model

wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json
wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params
mv Inception-BN-0126.params Inception-BN-0000.params

cd ...
```

We will also need to download the ImageNet categories which are essentially a mapping from integer classes to human readable strings.

```bash
$ wget http://data.mxnet.io/models/imagenet/synset.txt

$ head -10 synset.txt
n01440764 tench, Tinca tinca
n01443537 goldfish, Carassius auratus
n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
n01491361 tiger shark, Galeocerdo cuvieri
n01494475 hammerhead, hammerhead shark
n01496331 electric ray, crampfish, numbfish, torpedo
n01498041 stingray
n01514668 cock
n01514859 hen
n01518878 ostrich, Struthio camelus
```

## Pretrained Models

Before we dive into the code, we can compare the different pretrained models that we have downloaded. Choosing the right model for your application comes down to understanding the tradeoffs and the choices you have.

* What is your RAM constraint for running the model?
* How fast does the model run on CPU, GPU?
* What is the error rate you can deal with?

Below is a table that summarizes some key metrics for the three different models we will be using:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width:100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-0pky{border-color:#ccc;border-width:1px;text-align:middle;vertical-align:top;font-family:bold; width:25%;font-weight:bold}
.tg .tg-0lax{text-align:right;vertical-align:top}
.tg .tg-1lax{text-align:center;vertical-align:middle;background-color:#f0f0f0;font-weight:bold}
.tg .tg-2lax{text-align:left}
</style>
<table class="tg">
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">VGG16</th>
    <th class="tg-0pky">Inception v3</th>
  </tr>
  <tr>
    <td class="tg-1lax">RAM required</td>
    <td class="tg-0lax">528 MB</td>
    <td class="tg-0lax">43 MB</td>
  </tr>
  <tr>
    <td class="tg-1lax">Number of Parameters</td>
    <td class="tg-0lax">140 million</td>
    <td class="tg-0lax">25 million</td>
  </tr>
  <tr>
    <td class="tg-1lax">Error Rate (ImageNet Challenge)</td>
    <td class="tg-0lax">7.4%</td>
    <td class="tg-0lax">5.6%</td>
  </tr>
  <tr>
    <td class="tg-1lax">Topology</td>
    <td class="tg-0lax">

      <a href="/assets/images/viz/vgg16.png" >
        <img class="small-image" src="/assets/images/viz/vgg16.png" alt="VGG16 Topology" />
      </a>

    </td>
    <td class="tg-0lax">

      <a href="/assets/images/viz/inception.png" >
        <img class="small-image" src="/assets/images/viz/inception.png" alt="Inception Topology" />
      </a>
    </td>
  </tr>
</table>

If you want to learn more about the **MXNet vizualization API**, it is covered in a previous post [here][8].

### VGG16

Winner of the 2014 [ImageNet challenge][1] by achieving a **7.4%** error rate on image classification. It is a model built from 16 layers - [Research Paper][2]

### Inception v3

Published in December 2015. It achieved **15-25%** more accurate predictions than the best models at the time while being six times cheaper computationally and using five time less parameters - [Research Paper][4]

## Performing image Classification with pretrained Models

### Loading the models

Now that the models are on your disk, MXNet can load their computation graphs and the pretrained parameters (weights and biases). First, we define where the models are located and the size of the images that we will feed into the Models.

```clojure
(def model-dir "model/")

(def h 224) ;; Image height
(def w 224) ;; Image width
(def c 3)   ;; Number of channels: Red, Green, Blue
```

Time has come to load the models with the MXNet Module API.

```clojure
;; Loading VGG16
(defonce vgg-16-mod
  (-> {:prefix (str model-dir "vgg16") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

;; Loading Inception v3
(defonce inception-mod
  (-> {:prefix (str model-dir "Inception-BN") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))
```

The last step is to load the ImageNet labels that we downloaded before.

```clojure
(defonce image-net-labels
  (-> (str model-dir "/synset.txt")
      (slurp)
      (string/split #"\n")))

;; ImageNet 1000 Labels check
(assert (= 1000 (count image-net-labels)))
```

### Preparing the Data

Our models expect the data to have the following shape `[batch-size c h w]` where
* `batch-size`: number of datapoints to feed - we will set `batch-size` to `1` here since we want to predict one image at a time and not batch them
* `c`: number of color channels - Red, Green, Blue - `3`
* `h`: height of the image - `244`
* `w`: width of the image - `244`

We also need to normalize the image pixel values according to the way the different models were trained. In our case, we only need to subtract the mean (from the ImageNet dataset) color channels which are well known values.

```clojure
(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat
      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Substract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))
      ;; Flatten matrix
      (cvu/mat->flat-rgb-array)
      ;; Reshape to (1, c, h, w)
      (ndarray/array [1 c h w])))
```

If you need a refresher on **image manipulation and processing**, it is covered in a previous post [here][7].

### Predicting

We would like to make new predictions on images that we will feed to the models. The three models return an `NDArray` of probabilities:
* 1000 values between `0.0` and `1.0`
* The sum of these 1000 values is equal to `1.0`
* Each index in the `NDArray` corresponds to a label (Eg. Dog, Cat, Guitar, ...)

We will look at the top k labels that the model returns for a given image.

```clojure
(defn- top-k
  "Return top `k` from prob-maps with :prob key"
  [k prob-maps]
  (->> prob-maps
       (sort-by :prob)
       (reverse)
       (take k)))

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `x`"
  ([model labels x]
   (predict model labels x 5))
  ([model labels x k]
   (let [probs (-> model
                   (m/forward {:data [x]})
                   (m/outputs)
                   (ffirst)
                   (ndarray/->vec))
         prob-maps (mapv (fn [p l] {:prob p :label l}) probs labels)]
     (top-k k prob-maps))))
```

To run the model, one only needs to use the `m/forward` function to perform a forward pass.

### Prediction Comparisons

We can now run the models on different images and look at the top 5 predictions to get a feel for how they perform.


<br/>

![Cat](/assets/images/cat-egyptian.jpg){: .center-image }
<figcaption class="caption">Cat</figcaption>

<br/>

```clojure
(->> "images/cat2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict inception-mod image-net-labels))
;({:prob 0.9669817, :label "n02124075 Egyptian cat"}
;{:prob 0.020066999, :label "n02123045 tabby, tabby cat"}
;{:prob 0.0071042357, :label "n02123159 tiger cat"}
;{:prob 0.005353994, :label "n02127052 lynx, catamount"}
;{:prob 4.658187E-5, :label "n02123597 Siamese cat, Siamese"})

(->> "images/cat2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict vgg-16-mod image-net-labels))
;({:prob 0.9030159, :label "n02124075 Egyptian cat"}
;{:prob 0.05147686, :label "n02123045 tabby, tabby cat"}
;{:prob 0.024212556, :label "n02123159 tiger cat"}
;{:prob 0.0099070445, :label "n02127052 lynx, catamount"}
;{:prob 3.7205187E-4, :label "n04040759 radiator"})

```

![Dog](/assets/images/dog-2.jpg){: .center-image }
<figcaption class="caption">Dog</figcaption>

<br/>

```clojure
(->> "images/dog2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict inception-mod image-net-labels))
;({:prob 0.7363852, :label "n02110958 pug, pug-dog"}
;{:prob 0.23988461, :label "n02108422 bull mastiff"}
;{:prob 0.013495497, :label "n02108915 French bulldog"}
;{:prob 0.0019004685, :label "n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"}
;{:prob 0.0013417465, :label "n04409515 tennis ball"})

(->> "images/dog2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict vgg-16-mod image-net-labels))
;({:prob 0.95628285, :label "n02110958 pug, pug-dog"}
;{:prob 0.02271582, :label "n02108422 bull mastiff"}
;{:prob 0.0075261267, :label "n02108915 French bulldog"}
;{:prob 0.0014686864, :label "n02086079 Pekinese, Pekingese, Peke"}
;{:prob 0.0012910544, :label "n02108089 boxer"})
```

![Guitar Player](/assets/images/guitarplayer.jpg){: .center-image }
<figcaption class="caption">Guitar Player</figcaption>

<br/>

```clojure
(->> "images/guitarplayer2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict inception-mod image-net-labels))
;({:prob 0.647201, :label "n03272010 electric guitar"}
;{:prob 0.3371953, :label "n04296562 stage"}
;{:prob 0.008809802, :label "n02676566 acoustic guitar"}
;{:prob 0.0024602208, :label "n02787622 banjo"}
;{:prob 0.0018765739, :label "n03759954 microphone, mike"})

(->> "images/guitarplayer2.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict vgg-16-mod image-net-labels))
;({:prob 0.73966444, :label "n03272010 electric guitar"}
;{:prob 0.105860166, :label "n04296562 stage"}
;{:prob 0.059584185, :label "n04141076 sax, saxophone"}
;{:prob 0.029627431, :label "n02787622 banjo"}
;{:prob 0.016049441, :label "n02676566 acoustic guitar"})
```

Overall the predictions look really good! Don't you think?

Below is a summary of the predictions to compare how the different models perform on those images.

<table class="tg">
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">VGG16</th>
    <th class="tg-0pky">Inception v3</th>
  </tr>
  <tr>
    <td class="tg-1lax" rowspan="5">
      <img alt="Dog" src="/assets/images/dog-2.jpg" />Dog
    </td>
    <td class="tg-2lax">1. pug, pug-dog</td>
    <td class="tg-2lax">1. pug, pug-dog</td>
  </tr>
  <tr>
    <td class="tg-2lax">2. bull mastiff</td>
    <td class="tg-2lax">2. bull mastiff</td>
  </tr>
  <tr>
    <td class="tg-2lax">3. French bulldog</td>
    <td class="tg-2lax">3. French bulldog</td>
  </tr>
  <tr>
    <td class="tg-2lax">4. Pekinese, Pekingese, Peke</td>
    <td class="tg-2lax">4. American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier</td>
  </tr>
  <tr>
    <td class="tg-2lax">5. boxer</td>
    <td class="tg-2lax">5. tennis ball</td>
  </tr>


  <tr>
    <td class="tg-1lax" rowspan="5">
      <img alt="Cat" src="/assets/images/cat-egyptian.jpg" />Cat
    </td>
    <td class="tg-2lax">1. Egyptian cat</td>
    <td class="tg-2lax">1. Egyptian cat</td>
  </tr>
  <tr>
    <td class="tg-2lax">2. tabby, tabby cat</td>
    <td class="tg-2lax">2. tabby, tabby cat</td>
  </tr>
  <tr>
    <td class="tg-2lax">3. tiger cat</td>
    <td class="tg-2lax">3. tiger cat</td>
  </tr>
  <tr>
    <td class="tg-2lax">4. lynx, catamount</td>
    <td class="tg-2lax">4. lynx, catamount</td>
  </tr>
  <tr>
    <td class="tg-2lax">5. radiator</td>
    <td class="tg-2lax">5. Siamese cat, Siamese</td>
  </tr>

  <tr>
    <td class="tg-1lax" rowspan="5">
      <img alt="Guitar Player" src="/assets/images/guitarplayer.jpg" />Guitar Player
    </td>
    <td class="tg-2lax">1. electric guitar</td>
    <td class="tg-2lax">1. electric guitar</td>
  </tr>
  <tr>
    <td class="tg-2lax">2. stage</td>
    <td class="tg-2lax">2. stage</td>
  </tr>
  <tr>
    <td class="tg-2lax">3. sax, saxophone</td>
    <td class="tg-2lax">3. acoustic guitar</td>
  </tr>
  <tr>
    <td class="tg-2lax">4. banjo</td>
    <td class="tg-2lax">4. banjo</td>
  </tr>
  <tr>
    <td class="tg-2lax">5. acoustic guitar</td>
    <td class="tg-2lax">5. microphone, mike</td>
  </tr>

</table>

## Conclusion

Now you can use pretrained, state of the art Deep Learning Models for image classification. Most of the code is about data preparation. The actual prediction code is only 4 lines of code.

## References and Resources

* [ImageNet Challenge][1]
* [VGG16 Research Paper][2]
* [Inception v3 Research Paper][4]
* [MXNet Model Zoo][5]
* [An Introduction to the MXNet API - Part 4][6]
* [MXNet made simple: Image Manipulation with OpenCV and MXNet][7]
* [MXNet made simple: Clojure Symbol Visualization API][8]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.pretrained
  "Tutorial on pretrained models with MXNet: Inception and VGG."
  (:require
    [clojure.string :as string]

    [opencv4.core :as cv]
    [opencv4.utils :as cvu]

    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.ndarray :as ndarray]))

;;; Loading the Models

(def model-dir "model/")

(def h 224) ;; Image height
(def w 224) ;; Image width
(def c 3)   ;; Number of channels: Red, Green, Blue

;; Pretrained Inception BN model loaded from disk
(defonce inception-mod
  (-> {:prefix (str model-dir "Inception-BN") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

(defonce vgg-16-mod
  (-> {:prefix (str model-dir "vgg16") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

;; ImageNet 1000 Labels

(defonce image-net-labels
  (-> (str model-dir "/synset.txt")
      (slurp)
      (string/split #"\n")))

(assert (= 1000 (count image-net-labels)))

;;; Preparing the Data

(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat
      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Substract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))
      ;; Flatten matrix
      (cvu/mat->flat-rgb-array)
      ;; Reshape to (1, c, h, w)
      (ndarray/array [1 c h w])))

;;; Predicting

(defn- top-k
  "Return top `k` from prob-maps with :prob key"
  [k prob-maps]
  (->> prob-maps
       (sort-by :prob)
       (reverse)
       (take k)))

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `x`"
  ([model labels x]
   (predict model labels x 5))
  ([model labels x k]
   (let [probs (-> model
                   (m/forward {:data [x]})
                   (m/outputs)
                   (ffirst)
                   (ndarray/->vec))
         prob-maps (mapv (fn [p l] {:prob p :label l}) probs labels)]
     (top-k k prob-maps))))

(comment
  (->> "images/guitarplayer.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.68194896, :label "n04296562 stage"}
  ;{:prob 0.06861413, :label "n03272010 electric guitar"}
  ;{:prob 0.04886661, :label "n10565667 scuba diver"}
  ;{:prob 0.044686787, :label "n03250847 drumstick"}
  ;{:prob 0.029348794, :label "n02676566 acoustic guitar"})

  (->> "images/guitarplayer.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.31067622, :label "n03272010 electric guitar"}
  ;{:prob 0.14873363, :label "n04296562 stage"}
  ;{:prob 0.04211086, :label "n04141076 sax, saxophone"}
  ;{:prob 0.032480247, :label "n04536866 violin, fiddle"}
  ;{:prob 0.022555437, :label "n03110669 cornet, horn, trumpet, trump"})

  (->> "images/guitarplayer2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.647201, :label "n03272010 electric guitar"}
  ;{:prob 0.3371953, :label "n04296562 stage"}
  ;{:prob 0.008809802, :label "n02676566 acoustic guitar"}
  ;{:prob 0.0024602208, :label "n02787622 banjo"}
  ;{:prob 0.0018765739, :label "n03759954 microphone, mike"})

  (->> "images/guitarplayer2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.73966444, :label "n03272010 electric guitar"}
  ;{:prob 0.105860166, :label "n04296562 stage"}
  ;{:prob 0.059584185, :label "n04141076 sax, saxophone"}
  ;{:prob 0.029627431, :label "n02787622 banjo"}
  ;{:prob 0.016049441, :label "n02676566 acoustic guitar"})

  (->> "images/cat.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.5226559, :label "n02119789 kit fox, Vulpes macrotis"}
  ;{:prob 0.14540964, :label "n02112018 Pomeranian"}
  ;{:prob 0.13845555, :label "n02119022 red fox, Vulpes vulpes"}
  ;{:prob 0.06784552, :label "n02120505 grey fox, gray fox, Urocyon cinereoargenteus"}
  ;{:prob 0.024868377, :label "n02441942 weasel"})

  (->> "images/cat.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.41937035, :label "n02119789 kit fox, Vulpes macrotis"}
  ;{:prob 0.26819462, :label "n02119022 red fox, Vulpes vulpes"}
  ;{:prob 0.07655225, :label "n02124075 Egyptian cat"}
  ;{:prob 0.049807232, :label "n02123159 tiger cat"}
  ;{:prob 0.034435965, :label "n02123045 tabby, tabby cat"})

  (->> "images/cat2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.9669817, :label "n02124075 Egyptian cat"}
  ;{:prob 0.020066999, :label "n02123045 tabby, tabby cat"}
  ;{:prob 0.0071042357, :label "n02123159 tiger cat"}
  ;{:prob 0.005353994, :label "n02127052 lynx, catamount"}
  ;{:prob 4.658187E-5, :label "n02123597 Siamese cat, Siamese"})

  (->> "images/cat2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.9030159, :label "n02124075 Egyptian cat"}
  ;{:prob 0.05147686, :label "n02123045 tabby, tabby cat"}
  ;{:prob 0.024212556, :label "n02123159 tiger cat"}
  ;{:prob 0.0099070445, :label "n02127052 lynx, catamount"}
  ;{:prob 3.7205187E-4, :label "n04040759 radiator"})

  (->> "images/dog.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.89285797, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.06376573, :label "n04409515 tennis ball"}
  ;{:prob 0.01919549, :label "n03942813 ping-pong ball"}
  ;{:prob 0.014978847, :label "n02108422 bull mastiff"}
  ;{:prob 0.0012790044, :label "n02808304 bath towel"})

  (->> "images/dog.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.96750915, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.01833086, :label "n02108422 bull mastiff"}
  ;{:prob 0.005593519, :label "n04409515 tennis ball"}
  ;{:prob 0.0017559915, :label "n02108089 boxer"}
  ;{:prob 8.5579534E-4, :label "n02096585 Boston bull, Boston terrier"})

  (->> "images/dog2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.7363852, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.23988461, :label "n02108422 bull mastiff"}
  ;{:prob 0.013495497, :label "n02108915 French bulldog"}
  ;{:prob 0.0019004685, :label "n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"}
  ;{:prob 0.0013417465, :label "n04409515 tennis ball"})

  (->> "images/dog2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.95628285, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.02271582, :label "n02108422 bull mastiff"}
  ;{:prob 0.0075261267, :label "n02108915 French bulldog"}
  ;{:prob 0.0014686864, :label "n02086079 Pekinese, Pekingese, Peke"}
  ;{:prob 0.0012910544, :label "n02108089 boxer"})
  )
```

[1]: http://image-net.org/challenges/LSVRC/
[2]: https://arxiv.org/abs/1409.1556
[3]: https://arxiv.org/abs/1512.03385
[4]: https://arxiv.org/abs/1512.00567
[5]: https://mxnet.incubator.apache.org/model_zoo/
[6]: https://towardsdatascience.com/an-introduction-to-the-mxnet-api-part-4-df22560b83fe
[7]: /mxnet-made-simple-image-manipulation/
[8]: /mxnet-made-simple-symbol-visualization/

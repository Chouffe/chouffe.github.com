---
title:  "MXNet made simple: Transfer Learning to achieve State of the Art on image classification tasks"
layout: post
date: 2019-08-17 00:00
image: /assets/images/mxnet-logo.png
published: true
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- transferlearning
- tutorial
star: true
category: blog
author: arthurcaillau
description: Leverage pretrained models to achieve state of the art on image classification tasks
---

This post will explain how transfer learning can make us achieve state of the art performance on the [Oxford-IIIT Dataset][2]. This technique is very common among practicioners as pretrained models can be leveraged to quickly learn new machine learning tasks.

## Transfer Learning

### What is Transfer Learning?

> Transfer Learning is a Machine Learning technique where a model trained for task **X** is reused for task **Y**.

### Why do we need Transfer Learning?

When task **X** and **Y** are similar in nature, it requires much less time and training data to learn **Y** when a pretrained model is available for task **X**. The learning of task **X** is transfered to the model that learns task **Y**.

> Eg. A model trained on classifying 1000 object categories (task **X**) can be leveraged to classify 35 pet breeds (task **Y**).

1. It is computationally and time expensive to retrain a model from scratch. It can take weeks on a GPU to train certain models to perform well
2. It is data expensive to train a model to perform a machine task from scratch
3. It requires knowledge or exploration of Deep Learning architectures that perform well on the task to learn

### How does Transfer learning work in practice?

Practicioners often pick a pretrained model that performs really well on task **X** and then fine tune the last layers of the network on the new task **Y**.

This technique requires very little data and very little training time to achieve close to State Of The Art (SOTA) results on the task **Y**.

## Oxford-IIIT Dataset

We will use the [Oxford-IIIT Dataset][2] to demonstrate how to perform transfer learning.

From the Oxford-IIIT Dataset website:

> A 37 category pet dataset with roughly 200 images for each class. The images have a large variation in scale, pose and lighting. Can also be used for localization.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width:100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-0pky{border-color:#ccc;border-width:1px;text-align:middle;vertical-align:top;font-family:bold; width:33%;}
.tg .tg-0lax{text-align:left;vertical-align:top}
.tg .tg-0lax img {height:120px}
</style>

<br/>
<table class="tg">
  <tr>
    <th class="tg-0pky">English Cocker Spaniel</th>
    <th class="tg-0pky">Russian Blue</th>
    <th class="tg-0pky">Pug</th>
  </tr>
  <tr>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/datasets/oxford-pet/english_cocker_spaniel_186.jpg" alt="English Cocker Spaniel" />
    </td>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/datasets/oxford-pet/Russian_Blue_116.jpg" alt="Russian Blue" />
    </td>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/datasets/oxford-pet/pug_69.jpg" alt="Pug" />
    </td>
  </tr>
</table>

This dataset has been explored more in depth in a [previous post][1] where the `im2rec` preprocessing tool is explained.
To download and preprocess the dataset with `im2rec` the following script should be run.
```bash
#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

data_path=$PROJECT_ROOT/data/oxford-pet/

[[ "${MXNET_HOME}" = ""  ]] && { echo >&2 "MXNET_HOME hast to be set to be able to use im2rec"; exit 1; }

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi

if [ ! -f "$data_path/saint_bernard/saint_bernard_33.jpg" ]; then

pushd $data_path

# Downloading the dataset
wget https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz
tar zxvf oxford-iiit-pet.tgz
rm oxford-iiit-pet.tgz
mv oxford-iiit-pet/images/* .
rm -rf oxford-iiit-pet
rm *.mat

# Organizing images into folders
for image in *jpg ; do
  label=`echo $image | awk -F_ '{gsub($NF,"");sub(".$", "");print}'`
  mkdir -p $label
  mv $image $label/$image
done

popd

fi

# Making .lst and .rec files for MXNet to load
if [ ! -f "$data_path/data_train2.lst" ]; then

# Cleaning up the images that are failing with OpenCV
rm -f $data_path/Abyssinian/Abyssinian_34.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_139.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_145.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_167.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_177.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_191.jpg

python $MXNET_HOME/tools/im2rec.py \
  --list \
  --train-ratio 0.8 \
  --recursive \
  $data_path/data $data_path

python $MXNET_HOME/tools/im2rec.py \
  --resize 224 \
  --center-crop \
  --num-thread 4 \
  $data_path/data $data_path

fi
```

In the rest of the post, I will assume the rec files have been properly generated.

## Before we begin...

We will need to import certain packages.

```clojure
(require '[clojure.string :as str]

;; MXNet namespaces
(require '[org.apache.clojure-mxnet.initializer :as init])
(require '[org.apache.clojure-mxnet.io :as mx-io])
(require '[org.apache.clojure-mxnet.module :as m])
(require '[org.apache.clojure-mxnet.eval-metric :as eval-metric])
(require '[org.apache.clojure-mxnet.ndarray :as ndarray])
(require '[org.apache.clojure-mxnet.symbol :as sym])
(require '[org.apache.clojure-mxnet.callback :as callback])
(require '[org.apache.clojure-mxnet.context :as context])

;; OpenCV namespaces
(require '[opencv4.mxnet :as mx-cv])
(require '[opencv4.core :as cv])
(require '[opencv4.utils :as cvu])))
```

## Performing Transfer Learing on the Oxford-IIIT Dataset

### Loading the dataset

First, let's define some parameters for loading the dataset and the pretrained model.

```clojure
(def batch-size 10)
(def data-shape [3 224 224])
(def train-rec "data/oxford-pet/data_train.rec")
(def valid-rec "data/oxford-pet/data_val.rec")
(def model-dir "model")
(def num-classes 37)
```

Then we load the dataset as a `RecordIter`.
```clojure
(defonce train-iter
  (mx-io/image-record-iter
    {:path-imgrec train-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape

     ;; Data Augmentation
     :shuffle true  ;; Whether to shuffle data randomly or not
     ; :max-rotate-angle 50  ;; Rotate by a random degree in [-50 50]
     ; :resize 300  ;; resize the shorter edge before cropping
     :rand-crop true  ;; randomely crop the image
     :rand-mirror true}))  ;; randomely mirror the image


;; ImageRecordIter for validation
(defonce val-iter
  (mx-io/image-record-iter
    {:path-imgrec valid-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape}))
```

### Loading the pretrained model

A base model has to be picked to demonstrate transfer learning. As we want to learn an image classification task, we can leverage a strong classification model that is known to perform well in practice.

`ResNet` models are trained on the [ImageNet dataset][4] to discriminate between more than 20000 categories of objects. Ranging from cats to musical instruments. They are well suited for transfer learning on a new image classification task.

We can start with a `ResNet18` which is small enough to get a baseline quickly - using a pretrained model for classification tasks is explained in a [previous blog post][3].

Run the following bash script to fetch a pretrained `ResNet18` model.

```bash
#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/resnet-18-0000.params" ]; then
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-symbol.json -P $model_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-0000.params -P $model_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt -P $model_path
fi
```

Now we can define a function that load in memory the network architecture and its learned parameters.

```clojure
(defn get-model!
  "Loads pretrained model given a `model-name`.

  Ex:
    (get-model! \"resnet-18\")"
  [model-name]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/" model-name) :epoch 0})]
    {:msymbol (m/symbol mod)
     :arg-params (m/arg-params mod)
     :aux-params (m/aux-params mod)}))
```

Next, we need to reuse the base model learned parameters. We remove the last fully connected layers of the `ResNet` model and plug a new one.

The idea is that the base layer has learned how to classify different shapes and patterns when the new classification head will learn how to combine this knowledge to discriminate between 37 pet breeds.

```clojure
(defn mk-fine-tune-model
  "Makes the fine tune symbol `net` given the pretrained network `msymbol`.

   `msymbol`: the pretrained network symbol
   `arg-params`: the argument parameters of the pretrained model
   `num-classes`: the number of classes for the fine-tune datasets
   `layer-name`: the layer name before the last fully-connected layer"
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            ;; Adding a classifier head to the base network `net`
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args (->> arg-params
                    (remove (fn [[k v]] (str/includes? k "fc1")))
                    (into {}))}))
```

### Training the model

Time has come to train our model. We define a function `fine-tune!` that does just that.

```clojure
(defn fit!
  "Trains the symbol `net` on `devs` with `train-iter` for training and
  `val-iter` for validation."
  [devs net arg-params aux-params num-epoch train-iter val-iter]
  (-> net
      ;; Converting the `net` symbol to a `module`
      (m/module {:contexts devs})
      ;; Binding data and labels for training
      (m/bind {:data-shapes (mx-io/provide-data-desc train-iter)
               :label-shapes (mx-io/provide-label-desc val-iter)})
      ;; Initializing parameters and auxiliary states
      (m/init-params {:arg-params arg-params
                      :aux-params aux-params
                      :allow-missing true})
      ;; Training the module
      (m/fit {:train-data train-iter
              :eval-data val-iter
              :num-epoch num-epoch
              :fit-params
              (m/fit-params
                {:eval-metric (eval-metric/accuracy)
                 :intializer (init/xavier {:rand-type "gaussian"
                                           :factor-type "in"
                                           :magnitude 2})
                 :batch-end-callback (callback/speedometer batch-size 10)})})))

(defn fine-tune!
  "Fine tunes `model` on `devs` for `num-epoch` with `train-iter` for training
  and `val-iter` for validation."
  ([model num-epoch devs]
   (fine-tune! model num-epoch devs train-iter val-iter))
  ([model num-epoch devs train-iter val-iter]
   (let [{:keys [msymbol arg-params aux-params]} model
         {:keys [net new-args]} (mk-fine-tune-model
                                  (assoc model :num-classes num-classes))]
     (fit! devs net new-args arg-params num-epoch train-iter val-iter))))
```

Now we can perform transfer learning on a CPU with the following code:
```clojure
;; Training with a ResNet18 base model on a CPU for 1 epochs
(fine-tune! (get-model! "resnet-18") 1 [(context/cpu)])
; 64% training accuracy in about 30min of training
; 87% validation accuracy
; Can feed on average 2.5 images per second to the model for training

; ...
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [540]       Speed: 2.43 samples/secTrain-accuracy=0.626063
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [550]       Speed: 2.33 samples/secTrain-accuracy=0.629401
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [560]       Speed: 2.09 samples/secTrain-accuracy=0.632799
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [570]       Speed: 2.37 samples/secTrain-accuracy=0.635201
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [580]       Speed: 2.20 samples/secTrain-accuracy=0.638210
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [590]       Speed: 2.39 samples/secTrain-accuracy=0.641117
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.64111674
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=2248485
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.87297297
```

> **Note**: It will be very slow and I recommend running this on a GPU to get a ~100x time speedup. I wrote a [tutorial][5] to explain how to get setup on a GPU with AWS.

As you can see, in **only 30 minutes** of training and **1 epoch**, our model achieves about **87%** accuracy on the classification task. It is already an excellent performance and it is achieved with very little data and very little time. I would challenge you to beat it and learn how to classify these 37 pet breeds in less than 30 minutes using the [Oxford-IIIT dataset][2] :p

This performance is a good baseline. Let's see if we can improve on it when training with more epochs.

```clojure
;; Training with a ResNet18 base model on a GPU for 6 epochs
(fine-tune! (get-model! "resnet-18") 6 [(context/gpu)])
; 91% training accuracy in less than 2min of training
; 91% validation accuracy
; Can feed on average 170 images per second to the model for training

; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [60]        Speed: 169.31 samples/sec       Train-accuracy=0.920338
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [70]        Speed: 168.73 samples/sec       Train-accuracy=0.919454
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [80]        Speed: 169.76 samples/sec       Train-accuracy=0.919560
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [90]        Speed: 170.03 samples/sec       Train-accuracy=0.918441
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.91796875
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=34549
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.91032606
```

Training for 6 epochs on a GPU results in **91% accuracy** on the validation set. It did not take longer than **2 minutes** on a GPU.

We can probably even get better performance by using a larger base model. Let's see how good it gets with a pretrained `ResNet50` this time.

```bash
#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/resnet-50-0000.params" ]; then
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json -P $model_path
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params -P $model_path
  wget http://data.mxnet.io/models/imagenet/resnet/synset.txt -P $model_path
fi
```

```clojure
;; Training with a ResNet50 base model on a GPU for 5 epochs
(fine-tune! (get-model! "resnet-50") 5 [(context/gpu)]))
; 98.1% training accuracy in less than 5min of training
; 94.4% validation accuracy
; SOTA 2018!!
; Can feed on average 49 images per second to the model for training

; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [10]        Speed: 48.33 samples/sec       Train-accuracy=0.071023
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [20]        Speed: 48.57 samples/sec       Train-accuracy=0.132440
; ...
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [170]       Speed: 48.66 samples/sec       Train-accuracy=0.981542
; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [180]       Speed: 47.69 samples/sec       Train-accuracy=0.981354
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.9814189
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=119747
; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.9436141
```

In less than **5 minutes** of training, our model achieves **94.4% accuracy** on the validation set. To give you an idea of how good that is, this level of performance was the State of the Art in 2018.

## Conclusion

Transfer Learning is a truly powerful Machine Learning technique. It can help us achieve State of the Art results on a new Machine Learning task.

We achieved SOTA 2018 on the Oxford-IIIT classification task in less than 5 minutes of training.

## References and Resources

* [MXNet made simple: Image RecordIO with im2rec and Data Loading][1]
* [Getting started with Clojure and MXNet on AWS][5]
* [MXNet made simple: Pretrained Models for image classification - Inception and VGG][3]
* [Oxford-IIIT Pet Dataset][2]
* [Wikipedia: ImageNet][4]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.finetune
  (:require [clojure.string :as str]

            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]

            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

;; Parameters
(def batch-size 10)
(def data-shape [3 224 224])
(def train-rec "data/oxford-pet/data_train.rec")
(def valid-rec "data/oxford-pet/data_val.rec")
(def model-dir "model")
(def num-classes 37)

;; ImageRecordIter for training
(defonce train-iter
  (mx-io/image-record-iter
    {:path-imgrec train-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape

     ;; Data Augmentation
     :shuffle true  ;; Whether to shuffle data randomly or not
     ; :max-rotate-angle 50  ;; Rotate by a random degree in [-50 50]
     ; :resize 300  ;; resize the shorter edge before cropping
     :rand-crop true  ;; randomely crop the image
     :rand-mirror true}))  ;; randomely mirror the image


;; ImageRecordIter for validation
(defonce val-iter
  (mx-io/image-record-iter
    {:path-imgrec valid-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape}))

(defn get-model!
  "Loads pretrained model given a `model-name`.

  Ex:
    (get-model! \"resnet-18\")"
  [model-name]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/" model-name) :epoch 0})]
    {:msymbol (m/symbol mod)
     :arg-params (m/arg-params mod)
     :aux-params (m/aux-params mod)}))

(defn mk-fine-tune-model
  "Makes the fine tune symbol `net` given the pretrained network `msymbol`.

   `msymbol`: the pretrained network symbol
   `arg-params`: the argument parameters of the pretrained model
   `num-classes`: the number of classes for the fine-tune datasets
   `layer-name`: the layer name before the last fully-connected layer"
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            ;; Adding a classifier head to the base network `net`
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args (->> arg-params
                    (remove (fn [[k v]] (str/includes? k "fc1")))
                    (into {}))}))

(defn fit!
  "Trains the symbol `net` on `devs` with `train-iter` for training and
  `val-iter` for validation."
  [devs net arg-params aux-params num-epoch train-iter val-iter]
  (-> net
      ;; Converting the `net` symbol to a `module`
      (m/module {:contexts devs})
      ;; Binding data and labels for training
      (m/bind {:data-shapes (mx-io/provide-data-desc train-iter)
               :label-shapes (mx-io/provide-label-desc val-iter)})
      ;; Initializing parameters and auxiliary states
      (m/init-params {:arg-params arg-params
                      :aux-params aux-params
                      :allow-missing true})
      ;; Training the module
      (m/fit {:train-data train-iter
              :eval-data val-iter
              :num-epoch num-epoch
              :fit-params
              (m/fit-params
                {:eval-metric (eval-metric/accuracy)
                 :intializer (init/xavier {:rand-type "gaussian"
                                           :factor-type "in"
                                           :magnitude 2})
                 :batch-end-callback (callback/speedometer batch-size 10)})})))

(defn fine-tune!
  "Fine tunes `model` on `devs` for `num-epoch` with `train-iter` for training
  and `val-iter` for validation."
  ([model num-epoch devs]
   (fine-tune! model num-epoch devs train-iter val-iter))
  ([model num-epoch devs train-iter val-iter]
   (let [{:keys [msymbol arg-params aux-params]} model
         {:keys [net new-args]} (mk-fine-tune-model
                                  (assoc model :num-classes num-classes))]
     (fit! devs net new-args arg-params num-epoch train-iter val-iter))))

(comment

  (require '[mxnet-clj-tutorials.finetune :refer :all])
  (require '[org.apache.clojure-mxnet.context :as context])

  ;; On CPU: will be very slow
  (fine-tune! (get-model! "resnet-18") 1 [(context/cpu)])
  ; 64% training accuracy in about 30min of training
  ; 87% validation accuracy
  ; Can feed on average 2.5 images per second to the model for training

  ; ...
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [540]       Speed: 2.43 samples/secTrain-accuracy=0.626063
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [550]       Speed: 2.33 samples/secTrain-accuracy=0.629401
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [560]       Speed: 2.09 samples/secTrain-accuracy=0.632799
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [570]       Speed: 2.37 samples/secTrain-accuracy=0.635201
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [580]       Speed: 2.20 samples/secTrain-accuracy=0.638210
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [590]       Speed: 2.39 samples/secTrain-accuracy=0.641117
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.64111674
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=2248485
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.87297297

  (fine-tune! (get-model! "resnet-18") 6 [(context/gpu)])
  ; 91% training accuracy in less than 2min of training
  ; 91% validation accuracy
  ; Can feed on average 170 images per second to the model for training

  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [60]        Speed: 169.31 samples/sec       Train-accuracy=0.920338
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [70]        Speed: 168.73 samples/sec       Train-accuracy=0.919454
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [80]        Speed: 169.76 samples/sec       Train-accuracy=0.919560
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [90]        Speed: 170.03 samples/sec       Train-accuracy=0.918441
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.91796875
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=34549
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.91032606

  (fine-tune! (get-model! "resnet-50") 5 [(context/gpu)]))
  ; On a bigger model: Resnet50
  ; 98.1% training accuracy in less than 5min of training
  ; 94.4% validation accuracy
  ; SOTA 2018!!
  ; Can feed on average 49 images per second to the model for training

  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [10]        Speed: 48.33 samples/sec       Train-accuracy=0.071023
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [20]        Speed: 48.57 samples/sec       Train-accuracy=0.132440
  ; ...
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [170]       Speed: 48.66 samples/sec       Train-accuracy=0.981542
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [180]       Speed: 47.69 samples/sec       Train-accuracy=0.981354
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.9814189
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=119747
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.9436141
```

[1]: /image-record-iter/
[2]: http://www.robots.ox.ac.uk/~vgg/data/pets/
[3]: /mxnet-made-simple-pretrained-models/
[4]: https://en.wikipedia.org/wiki/ImageNet
[5]: /mxnet-clojure-aws/

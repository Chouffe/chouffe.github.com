---
title:  "Mxnet made simple: Clojure Module API"
layout: post
date: 2019-03-01 10:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- module
star: true
category: blog
author: arthurcaillau
description: The Module API for training Neural Networks and making new predictions
---

In a [previous post][1], we talked about the **Symbol API**. Computation Graphs in mxnet are used to define the Neural Network topoligies. The **Module API** is used for training models and making new predictions.

We will follow these steps:

1. Generate a fake data set
2. Design a Computation Graph for the model
3. Train the model
4. Validate the model
5. Save the model

The goal is to design, train and run a model used for a classification task.

### Generating a Data Set

* Data Set size: **1000 datapoints**
* A datapoint contains **100 features**
* A feature will be a **float in `[0, 1]`**
* There are **10 different categories** and the network will have to learn the correct category for the given datapoint
* Training and validation sets will be split `80/20`

```clojure
;; Parameters for the generated dataset
(def sample-size 1000)
(def train-size 800)
(def valid-size (- sample-size train-size))
(def feature-count 100)
(def category-count 10)

;;; Generating the Data Set: X and Y
(def X
  (random/uniform 0 1 [sample-size feature-count]))

(def Y
  (-> sample-size
      (repeatedly #(rand-int category-count))
      (ndarray/array [sample-size])))
```

We can check whether the generated data set is correct

```clojure
;; Checking X and Y data
(ndarray/shape-vec X) ;[1000 100]
(take 10 (ndarray/->vec X)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)

(ndarray/shape-vec Y) ;[1000]
(take 10 (ndarray/->vec Y)) ;(2.0 0.0 8.0 2.0 7.0 9.0 1.0 0.0 0.0 5.0)
```

Now we can split the data `80/20`. We would normally shuffle the `X` and `Y` sets to avoid potential bias in the data distribution.

```clojure
;;; Splitting the Data Set in train/valid - 80/20
(def X-train
  (ndarray/crop X
                (mx-shape/->shape [0 0])
                (mx-shape/->shape [train-size feature-count])))

(def X-valid
  (ndarray/crop X
                (mx-shape/->shape [train-size 0])
                (mx-shape/->shape [sample-size feature-count])))

(def Y-train
  (ndarray/crop Y
                (mx-shape/->shape [0])
                (mx-shape/->shape [train-size])))

(def Y-valid
  (ndarray/crop Y
                (mx-shape/->shape [train-size])
                (mx-shape/->shape [sample-size])))
```

It is good practice to check whether the sets `X-train`, `X-valid`, `Y-train` and `Y-valid` are correctly split

```clojure
;; Checking train and valid data
(ndarray/shape-vec X-train) ;[800 100]
(take 10 (ndarray/->vec X-train)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
(ndarray/shape-vec X-valid) ;[200 100]
(take 10 (ndarray/->vec X-valid)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
(ndarray/shape-vec Y-train) ;[800]
(take 10 (ndarray/->vec Y-train)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)
(ndarray/shape-vec Y-valid) ;[200]
(take 10 (ndarray/->vec Y-valid)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)
```

### Designing a Computation Graph for the model

Now it is time to describe a simple Neural Network model as a computation graph. We will use two fully connected layers with an activation function in between and a softmax layer to perform the classification task.

```clojure
(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden category-count})
    (sym/softmax-output "softmax" {:data data})))
```

We can also generate te Computation Graph of the Model to understand what it does

![Computation Graph of the Model](/assets/images/mlp.png){: .center-image }
<figcaption class="caption">Computation Graph of the Neural Network Model</figcaption>

### Training the model

#### Building the Data Iterator

Before training the model, we need a way to iterate over the data sets that we have generated earlier. Mxnet provides an abstaction called `NDIter` that lets us just do that.

```clojure
;;; Building the Data Iterator
(def train-iter
  (mx-io/ndarray-iter [X-train]
                      {:label-name "softmax_label"
                       :label [Y-train]
                       :data-batch-size batch-size}))

(def valid-iter
  (mx-io/ndarray-iter [X-valid]
                      {:label-name "softmax_label"
                       :label [Y-valid]
                       :data-batch-size batch-size}))
```

#### Time for training

We only need to wrap the computation graph define above in a `module`
```clojure
;; Wrapping the computation graph in a `module`
(def model-module (m/module (get-symbol)))
```

Finally, one needs to define the following parameters to run the training algorithm
* **Weight initializer** - we choose `Xavier` initializer
* **Optimizer Algorithm** - here we pick `SGD` (Stochastic Gradient Descent)
* **Learning Rate** - we choose a learning rate of `0.1`
* **Number of Epochs** - Number of times the model will run the entire training set during its training

```clojure
;;; Training the Model
(defn train! [model-module]
  (-> model-module
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label train-iter)})
      ;; Initializing weights with Xavier
      (m/init-params {:initializer (initializer/xavier)})
      ;; Choosing Optimizer Algorithm: SGD with lr = 0.1
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      ;; Training for `num-epochs`
      (m/fit {:train-data train-iter :eval-data valid-iter :num-epoch 50})))

(train! model-module)
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.105
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=275
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Train-accuracy=0.12125
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Time cost=154
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Train-accuracy=0.14625
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Time cost=134
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Train-accuracy=0.165
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Time cost=135
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[4] Train-accuracy=0.19875
; ...
; ...
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Validation-accuracy=0.085
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Time cost=126
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Validation-accuracy=0.09

;; Wow! Training accuracy is 1.0 -> It got 100% of training data right!
```

The model achieves 100% accuracy on the training set.

### Validating the Model

We always need to validate the model accuracy on a data set that has not been seen during training. This way, we can understand how the model will behave with unseen datapoints - It allows us to measure how it can generalize.

```clojure
;;; Validating the Model
(m/score model-module
         {:eval-data valid-iter
          :eval-metric (eval-metric/accuracy)}) ;["accuracy" 0.09]

;; Really bad!
;; Of course the model cannot generalize because it is random data!
;; The data is completely meaningless.
;; The model will not be able to predict anything!
```

We get a 9% accuracy which is worse than random choice for a 10 category classifier. It is not really surprising given the fact that the dataset has been generated randomly: there is nothing valuable to learn in it! Data is garbage!

### Saving the model

Training a model can take days/weeks. It is important to be able to save the model to disk.

```clojure
;;; Saving the Model to disk
(def save-prefix "my-model")

(m/save-checkpoint model-module
                   {:prefix save-prefix
                    :epoch 50
                    :save-opt-states true})

(def model-module-2
  (m/load-checkpoint {:prefix save-prefix
                      :epoch 50
                      :load-optimizer-states true}))

;; One can now resume training or start predicting with `model-module-2`
```

## Conclusion

The **Module API** makes it simple to train models and also make new predictions. Now that you have mastered the **NDArray API**, the **Symbol API** and the **Module API**, you can start building your own Deep Learning Models.

## References and Resources

* [Mxnet official Module API tutorial][3]
* [Module API Reference][4]
* [An introduction to the MXNet API — part 3][2]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.module
  "Tutorial for the `module` API."
  (:require
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.eval-metric :as eval-metric]
    [org.apache.clojure-mxnet.executor :as executor]
    [org.apache.clojure-mxnet.initializer :as initializer]
    [org.apache.clojure-mxnet.io :as mx-io]
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.optimizer :as optimizer]
    [org.apache.clojure-mxnet.random :as random]
    [org.apache.clojure-mxnet.shape :as mx-shape]
    [org.apache.clojure-mxnet.symbol :as sym]))

;; Inspiration from: https://mxnet.incubator.apache.org/api/clojure/module.html
;; The Module API lets us train/optimize a Neural Network symbol

(def sample-size 1000)
(def train-size 800)
(def valid-size (- sample-size train-size))

(def feature-count 100)
(def category-count 10)
(def batch-size 10)

;;; Generating the Data Set

(def X
  (random/uniform 0 1 [sample-size feature-count]))

(def Y
  (-> sample-size
      (repeatedly #(rand-int category-count))
      (ndarray/array [sample-size])))

;; Checking X and Y data

(ndarray/shape-vec X) ;[1000 100]
(take 10 (ndarray/->vec X)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)

(ndarray/shape-vec Y) ;[1000]
(take 10 (ndarray/->vec Y)) ;(2.0 0.0 8.0 2.0 7.0 9.0 1.0 0.0 0.0 5.0)

;;; Splitting the Data Set in train/valid - 80/20

(def X-train
  (ndarray/crop X
                (mx-shape/->shape [0 0])
                (mx-shape/->shape [train-size feature-count])))

(def X-valid
  (ndarray/crop X
                (mx-shape/->shape [train-size 0])
                (mx-shape/->shape [sample-size feature-count])))

(def Y-train
  (ndarray/crop Y
                (mx-shape/->shape [0])
                (mx-shape/->shape [train-size])))

(def Y-valid
  (ndarray/crop Y
                (mx-shape/->shape [train-size])
                (mx-shape/->shape [sample-size])))

;; Checking train and valid data

(ndarray/shape-vec X-train) ;[800 100]
(take 10 (ndarray/->vec X-train)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
(ndarray/shape-vec X-valid) ;[200 100]
(take 10 (ndarray/->vec X-valid)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
(ndarray/shape-vec Y-train) ;[800]
(take 10 (ndarray/->vec Y-train)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)
(ndarray/shape-vec Y-valid) ;[200]
(take 10 (ndarray/->vec Y-valid)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)

;;; Building the Network as a symbolic graph of computations

(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden category-count})
    (sym/softmax-output "softmax" {:data data})))

;;; Building the Data Iterator

(def train-iter
  (mx-io/ndarray-iter [X-train]
                      {:label-name "softmax_label"
                       :label [Y-train]
                       :data-batch-size batch-size}))

(def valid-iter
  (mx-io/ndarray-iter [X-valid]
                      {:label-name "softmax_label"
                       :label [Y-valid]
                       :data-batch-size batch-size}))

(def model-module (m/module (get-symbol)))

;;; Training the Model

(defn train! [model-module]
  (-> model-module
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label train-iter)})
      ;; Initializing weights with Xavier
      (m/init-params {:initializer (initializer/xavier)})
      ;; Choosing Optimizer Algorithm: SGD with lr = 0.1
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      ;; Training for `num-epochs`
      (m/fit {:train-data train-iter :eval-data valid-iter :num-epoch 50})))

(train! model-module)
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.105
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=275
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Train-accuracy=0.12125
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Time cost=154
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Train-accuracy=0.14625
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Time cost=134
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Train-accuracy=0.165
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Time cost=135
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[4] Train-accuracy=0.19875
; ...
; ...
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Validation-accuracy=0.085
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Time cost=126
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Validation-accuracy=0.09

;; Wow! Training accuracy is 1.0 -> It got 100% of training data right!

;;; Validating the Model

(m/score model-module
         {:eval-data valid-iter
          :eval-metric (eval-metric/accuracy)}) ;["accuracy" 0.09]

;; Really bad!
;; Of course the model cannot generalize because it is random data!
;; The data is completely meaningless.
;; The model will not be able to predict anything!

;;; Saving the Model to disk

(def save-prefix "my-model")

(m/save-checkpoint model-module
                   {:prefix save-prefix
                    :epoch 50
                    :save-opt-states true})

(def model-module-2
  (m/load-checkpoint {:prefix save-prefix
                      :epoch 50
                      :load-optimizer-states true}))

;; One can now resume training or start predicting with `model-module-2`
```

[1]: /mxnet-made-simple-symbol-api/
[2]: https://medium.com/@julsimon/an-introduction-to-the-mxnet-api-part-3-1803112ba3a8
[3]: https://mxnet.incubator.apache.org/api/clojure/module.html
[4]: https://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.module.html

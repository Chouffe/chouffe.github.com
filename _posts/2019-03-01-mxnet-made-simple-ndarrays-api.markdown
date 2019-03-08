---
title:  "MXNet made simple: Clojure NDArray API"
layout: post
date: 2019-03-01 00:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- ndarray
star: true
category: blog
author: arthurcaillau
description: MXNet NDArray is the fundamental data structure used for Symbolic Computations
---

An `NDArray` is an n-dimensional array that contains numerical values of identical type and size. They are called **tensors**. `NDArrays` generalize the concept of scalars, vectors and matrices.

Understanding `NDArray` is critical since it is the data structure that will be used to perform neural network operations. A Neural Network is nothing more than a function that takes in `NDArray`s and returns `NDArray`s performing some tensor computations that will be described in a following post. In this post, I will walk you through how we can create `NDArray`s and perform simple operations on them  using `Clojure MXNet`.

```haskell
NeuralNetwork :: [NDArray] -> [NDArray]
```

##### Notes

[Apache MXNet][1] is a open-source Deep Learning framework that is an alternative to Google's [Tensorflow][2]. A Neural Network is a description of how **tensors** flow!

If you are familiar with `Python`, the [numpy][3] library provides a very similar abstraction.

### Before we begin...

We will need to import certain packages:
```clojure
(require '[org.apache.clojure-mxnet.ndarray :as ndarray])
(require '[org.apache.clojure-mxnet.dtype :as d])
```

### Basic Operations

#### Creation

With specific values
```clojure
;; Create an `ndarray` with content set to a specific value
(def a (ndarray/array [1 2 3 4 5 6] [2 3]))

;; Getting the shape as a Clojure vector
(ndarray/shape-vec a) ;[2 3]

;; Visualizing the ndarray as a Clojure vector
(ndarray/->vec a) ;[1.0 2.0 3.0 4.0 5.0 6.0]
```

With `0s` or `1s` values
```clojure
(let [b (ndarray/zeros [2 5])
      c (ndarray/ones [1 3 24 24])]
  (println (ndarray/shape-vec b)) ;[2 5]
  (println (ndarray/->vec b)) ;[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
  (println (ndarray/shape-vec c)) ;[1 3 24 24]
  (println (ndarray/->vec c)) ;[1.0 1.0 ... 1.0]
  )
```

#### Set NDArray Type

An `NDArray` holds 32-bit floats by default
```clojure
(def a (ndarray/array [1 2 3 4 5 6] [2 3]))

;; Getting the dtype
(ndarray/dtype a) ;#object[scala.Enumeration$Val 0x781578fc "float32"]
```
Its type can be change to another numeric type
```clojure
(let [b (ndarray/as-type a d/INT32)]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x7364f96f int32]
  (println (ndarray/->vec b)) ;[1.0 2.0 3.0 4.0 5.0 6.0]
  )
```
Here is the list of `dtype`s available in MXNet:

* `UINT8`
* `INT32`
* `FLOAT16`
* `FLOAT32`
* `FLOAT64`

### Tensor Operations

MXNet supports a lot of tensor operations. Only a few will be discussed in this post. You can look at the [NDArray API reference][4] when needed.

#### Arithmetic Operations

All the basic arithmetic operations you know are available on `NDArrays`: `+`, `-`, `*`, `\`, etc.

```clojure
(let [b (ndarray/ones [1 5])
      c (ndarray/zeros [1 5])]
  (println (ndarray/->vec (ndarray/+ b c))) ;[1.0 1.0 1.0 1.0 1.0]
  (println (ndarray/->vec (ndarray/* b c))) ;[0.0 0.0 0.0 0.0 0.0]
  )
```

#### Slice Operations

Slicing `NDArrays` is really useful in the case of images. For instance, we could slice the red pixel values from the `NDArray` representing an RGB image.

```clojure
;; Slice Operations
(let [b (ndarray/array (range 1 (inc (* 3 2 2))) [3 2 2]) ; Pretend b is an RGB image of size 2x2
      b0 (ndarray/slice b 0) ;; Retrieving the red pixel values
      b1 (ndarray/slice b 1)  ;; Retrieving the green pixel values
      b2 (ndarray/slice b 1 3)] ;; Retrieving the green and blue pixel values

  (println (ndarray/->vec b0)) ;[1.0 2.0 3.0 4.0]
  (println (ndarray/shape-vec b0)) ;[1 2 2]

  (println (ndarray/->vec b1)) ;[5.0 6.0 7.0 8.0]
  (println (ndarray/shape-vec b1)) ;[1 2 2]

  (println (ndarray/->vec b2)) ;[5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
  (println (ndarray/shape-vec b2)) ;[2 2 2]
  )
```

#### Transposition

Transposing `NDArrays` in an efficient way is important when carrying out matrix operations in a Neural Network

```clojure
(let [at (ndarray/transpose a)]
  (println (ndarray/shape-vec a)) ;[2 3]
  (println (ndarray/shape-vec at)) ;[3 2]
  )
```

#### Dot Product

The famous **Dot Product** is also available for `NDArrays`. Very useful when designing evaluation metrics for instance.

```clojure
(let [b (ndarray/transpose a)
      c (ndarray/dot a b)
      d (ndarray/dot b a)]
  (println (ndarray/->vec c)) ;[14.0 32.0 32.0 77.0]
  (println (ndarray/shape-vec c)) ;[2 2]
  (println (ndarray/->vec d)) ;[17.0 22.0 27.0 22.0 29.0 36.0 27.0 36.0 45.0]
  (println (ndarray/shape-vec d)) ;[3 3]
  )
```

### Random Initialization

A very important idea in Machine Learning is randomness. We will see later on that we need to randomly initialize some parameters in the models before training them.

```clojure
(let [u (ndarray/uniform 0 1 (mx-shape/->shape [2 2]))
      g (ndarray/normal 0 1 (mx-shape/->shape [2 2]))]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )
```

MXNet also provides a namespace named `random` containing the following probability distributions to initialize `NDArrays`

```clojure
(require '[org.apache.clojure-mxnet.random :as random])

;; Initializing random ndarrays with `random`
(let [u (random/uniform 0 1 [2 2])
      g (random/normal 0 1 [2 2])]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )
```

## Data Processing

Machine Learning Models are trained on images, texts, videos, etc.
The data needs to be converted into `NDArray`s for the models to use it. This processing step is necessary to train the models and make new predictions.


The next section will detail how one would do it with images.

#### MNIST Dataset

The MNIST database is a large database of handwritten digits that is used to train machine learning systems.
One image digit from this database is shown below

![Digit 8](/assets/images/mnist-8.png){: .center-image }
<figcaption class="caption">Digit 8</figcaption>

This digit is a 2-dimensional vector containing pixel values in the range `[0, 255]`. This is how grayscale images are stored.

![Digit 8 pixel values](/assets/images/mnist-8-values.png){: .center-image }
<figcaption class="caption">2-dimensional vector of pixel values</figcaption>

An image from this Dataset can easily be converted into an `NDArray` and then used by the models.
MXNet provides an `image` namespace for loading image data into `NDArrays`.

```clojure
(require '[org.apache.clojure-mxnet.image :as mx-img])

;;; Loading Data as NDArrays
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd (mx-img/read-image img-filename {:color-flag 0})
      img-color-nd (mx-img/read-image img-filename {:color-flag 1})]
  ;; Grayscale image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )
```

The code above demonstrates how to read images from disk and turn them into `NDArrays`.
The `color-flag` parameter in the `mx-img/read-image` function tells MXNet to load either the image in grayscale (only one channel) or with colors (three channels).

Under the hood, MXNet uses **OpenCV** to read images from disk. One can directly use OpenCV to do it

```clojure
(require '[opencv4.core :as cv])
(require '[opencv4.utils :as cvu])

;; With OpenCV
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd  (-> img-filename
                            (cv/imread cv/COLOR_BGR2GRAY)
                            cvu/mat->flat-rgb-array
                            (ndarray/array [28 28 1]))
      img-color-nd (-> img-filename
                       cv/imread
                       cvu/mat->flat-rgb-array
                       (ndarray/array [28 28 3]))]
  ;; Grayscale Image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 float]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )
```

## Conclusion

`NDArrays` are the core data-structures used in MXNet. Understanding what they are and what one can do with them is fundamental.

Being able to transform your raw data (images, text, videos, ...) into `NDArray` is an important skill because everything that will flow into your models will be `NDArrays`.

This post should give you the necessary background needed to understand symbolic computations that will be covered in the next post.

## References and Resources

* [MXNet Website][1]
* [Tensorflow Website][2]
* [Numpy Website][3]
* [NDArray API Reference][4]
* [MXNet official NDArray API tutorial][6]
* [MXNet Github Repo][5]
* [An introduction to the MXNet API — part 1][7]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.ndarray
  "Tutorial for `ndarray` manipulations"
  (:require
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.image :as mx-img]
    [org.apache.clojure-mxnet.random :as random]
    [org.apache.clojure-mxnet.shape :as mx-shape]

    [opencv4.core :as cv]
    [opencv4.utils :as cvu]))

;; Create an `ndarray` with content set to a specific value
(def a
  (ndarray/array [1 2 3 4 5 6] [2 3]))

;; Getting the dtype
(ndarray/dtype a) ;#object[scala.Enumeration$Val 0x781578fc "float32"]

;; Getting the shape as a clojure vector
(ndarray/shape-vec a) ;[2 3]

;; Visualizing the ndarray as a Clojure vector
(ndarray/->vec a) ;[1.0 2.0 3.0 4.0 5.0 6.0]

;; Ndarray creations
(let [b (ndarray/zeros [2 5])
      c (ndarray/ones [1 3 24 24])]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x781578fc float32]
  (println (ndarray/shape-vec b)) ;[2 5]
  (println (ndarray/->vec b)) ;[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
  (println (ndarray/dtype c)) ;#object[scala.Enumeration$Val 0x781578fc float32]
  (println (ndarray/shape-vec c)) ;[1 3 24 24]
  (println (ndarray/->vec c)) ;[1.0 1.0 ... 1.0]
  )

;; Cast to another dtype
(let [b (ndarray/as-type a d/INT32)]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x7364f96f int32]
  (println (ndarray/->vec b)) ;[1.0 2.0 3.0 4.0 5.0 6.0]
  )

;; Arithmetic Operations
(let [b (ndarray/ones [1 5])
      c (ndarray/zeros [1 5])]
  (println (ndarray/->vec (ndarray/+ b c))) ;[1.0 1.0 1.0 1.0 1.0]
  (println (ndarray/->vec (ndarray/* b c))) ;[0.0 0.0 0.0 0.0 0.0]
  )

(let [b (ndarray/array (range 1 (inc (* 3 3 3))) [3 3 3])]
  (ndarray/shape-vec (ndarray/slice b 1))
  )

;; Slice Operations
(let [b (ndarray/array (range 1 (inc (* 3 2 2))) [3 2 2]) ; Pretend b is an RGB image of size 2x2
      b0 (ndarray/slice b 0) ;; Retrieving the red pixel values
      b1 (ndarray/slice b 1)  ;; Retrieving the green pixel values
      b2 (ndarray/slice b 1 3)] ;; Retrieving the green and blue pixel values

  (println (ndarray/->vec b0)) ;[1.0 2.0 3.0 4.0]
  (println (ndarray/shape-vec b0)) ;[1 2 2]

  (println (ndarray/->vec b1)) ;[5.0 6.0 7.0 8.0]
  (println (ndarray/shape-vec b1)) ;[1 2 2]

  (println (ndarray/->vec b2)) ;[5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
  (println (ndarray/shape-vec b2)) ;[2 2 2]
  )

;; Transposition
(let [at (ndarray/transpose a)]
  (println (ndarray/shape-vec a)) ;[2 3]
  (println (ndarray/shape-vec at)) ;[3 2]
  )

;; Matrix operations
(let [b (ndarray/transpose a)
      c (ndarray/dot a b)
      d (ndarray/dot b a)]
  (println (ndarray/->vec c)) ;[14.0 32.0 32.0 77.0]
  (println (ndarray/shape-vec c)) ;[2 2]
  (println (ndarray/->vec d)) ;[17.0 22.0 27.0 22.0 29.0 36.0 27.0 36.0 45.0]
  (println (ndarray/shape-vec d)) ;[3 3]
  )

;; Initializing random ndarrays
(let [u (ndarray/uniform 0 1 (mx-shape/->shape [2 2]))
      g (ndarray/normal 0 1 (mx-shape/->shape [2 2]))]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )

;; Initializing random ndarrays with `random`
(let [u (random/uniform 0 1 [2 2])
      g (random/normal 0 1 [2 2])]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )

;;; Loading Data as NDArrays

;; With mxnet `image` API
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd (mx-img/read-image img-filename {:color-flag 0})
      img-color-nd (mx-img/read-image img-filename {:color-flag 1})]
  ;; Grayscale image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )

;; With OpenCV
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd  (-> img-filename
                            (cv/imread cv/COLOR_BGR2GRAY)
                            cvu/mat->flat-rgb-array
                            (ndarray/array [28 28 1]))
      img-color-nd (-> img-filename
                       cv/imread
                       cvu/mat->flat-rgb-array
                       (ndarray/array [28 28 3]))]
  ;; Grayscale Image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 float]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )

;; Note:
;; * OpenCV default for channel ordering is `BGR`
;; * mxnet default for channel ordering is `RGB`
```

[1]: https://mxnet.apache.org/
[2]: https://www.tensorflow.org/
[3]: https://www.numpy.org/
[4]: https://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.ndarray.html
[5]: https://github.com/apache/incubator-mxnet
[6]: https://mxnet.incubator.apache.org/api/clojure/ndarray.html
[7]: https://becominghuman.ai/an-introduction-to-the-mxnet-api-part-1-848febdcf8ab

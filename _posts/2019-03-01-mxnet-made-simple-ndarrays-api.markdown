---
title:  "Mxnet made simple: NDarray API"
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
description: mxnet NDArray is the fundamental data structure used for Symbolic Computations
---

An `NDarray` is an n-dimensional array that contains numerical values of identical type and size. They are called **tensors**. `NDarrays` generalize the concept of scalars and vectors.

Understanding `NDarray` is critical since it is the data structure that will be used to perform neural network operations. A Neural Network is nothing more than a function that takes in `Ndarrays` and returns `Ndarrays` performing some tensor computations that will be described in an other post.

```haskell
NeuralNetwork :: [Ndarray] -> [NDarray]
```

##### Notes

If you are familiar with `Python`, the [numpy][3] library provides a very similar abstraction.

[Tensorflow][2] is the Deep Learning framework of Google. A Neural Network is a description of how **tensors** flow!

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

#### Set NDarray Type

An `NDArray` holds 32-bit floats by default
```clojure
(def a (ndarray/array [1 2 3 4 5 6] [2 3]))

;; Getting the dtype
(ndarray/dtype a) ;#object[scala.Enumeration$Val 0x781578fc "float32"]
```
It can be changed to another numeric type
```clojure
(let [b (ndarray/as-type a d/INT32)]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x7364f96f int32]
  (println (ndarray/->vec b)) ;[1.0 2.0 3.0 4.0 5.0 6.0]
  )
```
Here is the list of `dtype` available in mxnet:

* `UINT8`
* `INT32`
* `FLOAT16`
* `FLOAT32`
* `FLOAT64`

### Tensor Operations

Mxnet supports a lot of tensor operations. Only a few will be discussed in this post. You can look at the [NDarray API reference](https://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.ndarray.html) when needed.

#### Arithmetic Operations

```clojure
(let [b (ndarray/ones [1 5])
      c (ndarray/zeros [1 5])]
  (println (ndarray/->vec (ndarray/+ b c))) ;[1.0 1.0 1.0 1.0 1.0]
  (println (ndarray/->vec (ndarray/* b c))) ;[0.0 0.0 0.0 0.0 0.0]
  )
```

#### Slice Operations

```clojure
(let [b (ndarray/array [1 2 3 4 5 6] [3 2])
      b1 (ndarray/slice b 1)
      b2 (ndarray/slice b 1 3)]

  (println (ndarray/->vec b1)) ;[3.0 4.0]
  (println (ndarray/shape-vec b1)) ;[1 2]

  (println (ndarray/->vec b2)) ;[3.0 4.0 5.0 6.0]
  (println (ndarray/shape-vec b2)) ;[2 2]
  )
```

#### Transposition

```clojure
(let [at (ndarray/transpose a)]
  (println (ndarray/shape-vec a)) ;[2 3]
  (println (ndarray/shape-vec at)) ;[3 2]
  )
```

#### Dot Product

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

Mxnet also provides a namespace named `random` containing the following random initialization of `NDArray`s

```clojure
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
The data needs to be converted into `NDArrays` for the models to use it.

But fear not, it is often very simple to do so. The next section will detail how one would do it with an image.

#### MNIST Dataset

The MNIST database is a large database of handwritten digits that is used to train machine learning systems.
One image digit from this database is shown below

![Digit 8](/assets/images/mnist-8.png){: .center-image }
<figcaption class="caption">Digit 8</figcaption>

This digit is a 2-dimensional vector containing pixel values in the range `[0, 255]`. This is how grayscale images are stored.

![Digit 8 pixel values](/assets/images/mnist-8-values.png){: .center-image }
<figcaption class="caption">2-dimensional vector of pixel values</figcaption>

An image from this Dataset can easily be converted into an `NDarray` and then used by the models.

## Conclusion

`NDArrays` are the core datastructures used in mxnet. Understanding what they are and what one can do with them is fundamental.

This post should give you the necessary background needed to understand symbolic computations that will be covered in the next post.

## References and Resources

* [NDarray API Reference][1]
* [Mxnet official NDArray API tutorial][6]
* [Tensorflow Website][2]
* [Numpy Website][3]
* [Mxnet Website][4]
* [Mxnet Github Repo][5]
* [An introduction to the MXNet API — part 1][7]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.ndarray
  "Tutorial for `ndarray` manipulations"
  (:require
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.random :as random]
    [org.apache.clojure-mxnet.shape :as mx-shape]))

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

;; Slice Operations
(let [b (ndarray/array [1 2 3 4 5 6] [3 2])
      b1 (ndarray/slice b 1)
      b2 (ndarray/slice b 1 3)]

  (println (ndarray/->vec b1)) ;[3.0 4.0]
  (println (ndarray/shape-vec b1)) ;[1 2]

  (println (ndarray/->vec b2)) ;[3.0 4.0 5.0 6.0]
  (println (ndarray/shape-vec b2)) ;[2 2]
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
```

[1]: https://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.ndarray.html
[2]: https://www.tensorflow.org/
[3]: https://www.numpy.org/
[4]: https://mxnet.incubator.apache.org/
[5]: https://github.com/apache/incubator-mxnet
[6]: https://mxnet.incubator.apache.org/api/clojure/ndarray.html
[7]: https://becominghuman.ai/an-introduction-to-the-mxnet-api-part-1-848febdcf8ab

---
title:  "MXNet made simple: Image Manipulation with OpenCV and MXNet"
layout: post
date: 2019-03-17 00:00
image: /assets/images/mxnet-logo.png
headerImage: true
tag:
- clojure
- mxnet
- deeplearning
- ndarray
- image
- opencv
- tutorial
star: true
category: blog
author: arthurcaillau
description: Image Manipulation with OpenCV and MXNet
---

In this post, we will learn how to perform some common image manipulation operations that are used for preprocessing or postprocessing image datasets. [OpenCV][2] and **MXNet** can manipulate images and `NDArrays` directly.

Nothing specific to **Deep Learning** will be covered in this post although this is probably something you need to learn before being able to build your own Computer Vision models.

### Before we begin...

We will need to import certain packages:

```clojure
(require '[clojure.java.io :as io])

;; MXNet namespaces
(require '[org.apache.clojure-mxnet.image :as mx-img])
(require '[org.apache.clojure-mxnet.ndarray :as ndarray])
(require '[org.apache.clojure-mxnet.shape :as mx-shape])

;; OpenCV namespaces
(require '[opencv4.colors.rgb :as rgb])
(require '[opencv4.mxnet :as mx-cv])
(require '[opencv4.core :as cv])
(require '[opencv4.utils :as cvu])

(import org.opencv.core.Mat)
(import java.awt.image.DataBufferByte)
```

## Downloading images from URLs

We can write a small util function that lets us download an image from a URL and save it locally - similar to **wget**.

```clojure
(defn download!
  "Download `uri` and store it in `filename` on disk"
  [uri filename]
  (with-open [in (io/input-stream uri)
              out (io/output-stream filename)]
    (io/copy in out)))
```

Now, one can download images from URLs

```clojure
;; Download a cat image from a `uri` and save it into `images/cat.jpg`
(download! "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" "images/cat.jpg")

;; Download a dog image from a `uri` and save it into `images/dog.jpg`
(download! "https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true" "images/dog.jpg")
```

<br/>

![Cat](/assets/images/cat-small.jpg){: .center-image }
<figcaption class="caption">Cat</figcaption>

<br/>

![Dog](/assets/images/dog-small.jpg){: .center-image }
<figcaption class="caption">Dog</figcaption>

## Plotting an image from disk

It is often really useful to plot an image that is loaded from a filename on disk

```clojure
(defn preview!
  "Preview image from `filename` and display it on the screen in a new window
   Ex:
    (preview! \"images/cat.jpg\")
    (preview! \"images/cat.jpg\" :h 300 :w 200)"
  ([filename]
   (preview! filename {:h 400 :w 400}))
  ([filename {:keys [h w]}]
   (-> filename
       (cv/imread)
       (cv/resize! (cv/new-size h w))
       (cvu/imshow))))
```

The function `preview!` lets you load images in memory and display them

```clojure
;; Preview an image from disk
(preview! "images/cat.jpg")

;; Preview with different size
(preview! "images/cat.jpg" {:h 300 :w 200})
```

## Image preprocessing

#### Resizing, Scaling, Normalizing

When working with image datasets, we always have to preprocess them. Below is a list of preprocessing steps that are commonly done:

* Subtract mean of a pixel value for Red, Green and Blue channels
* Resize an image to fit the expected dimensions of a model
* Rescale pixel values linearly from a range to another - Eg. `(-128, 127) -> (0, 127)`
* Normalize pixel values into the `(0.0, 1.0)` range

```clojure
(defn preprocess-mat
  "Preprocessing steps on a `mat` from OpenCV.
   Example of commons preprocessing tasks"
  [mat]
  (-> mat
      ;; Subtract mean
      (cv/add! (cv/new-scalar 103.939 116.779 123.68))
      ;; Resize
      (cv/resize! (cv/new-size 400 400))
      ;; Maps pixel values from [-128, 127] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Normalize pixel values into (0.0, 1.0)
      ; (cv/normalize! 0.0 1.0 cv/NORM_MINMAX cv/CV_32FC1)
      ))
```

One can now visualize what the processing steps do to an image

```clojure
(-> "images/cat.jpg"
    (cv/imread)
    (preprocess-mat)
    (cvu/imshow))
```

<br/>

![Preprocessed Cat](/assets/images/cat-preprocessed.jpg){: .center-image }
<figcaption class="caption">Preprocessed Cat</figcaption>

#### Raw Byte Values

[Origami][3], the Clojure OpenCV wrapper, provides serveral utility functions to make it simple to look at raw byte values. One would need to do this when performing preprocessing steps in order to check that is was done properly at the pixel value level.

```clojure
;; Looking at the raw bytes
(-> "images/dog.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb false})
    ;; Resizing image
    (mx-img/resize-image 200 200)
    ;; Convert NDArray to Mat
    (mx-cv/ndarray-to-mat)
    ;; Mat to bytes
    (cv/<<)
    (cv/->bytes)
    (vec))
```

* `cv/<<`: converts the `mat` to bytes
* `cv/->bytes` returns the byte values from the mat as a `byte-array`

#### Color channels

A color image is represented as a matrix of pixel values for each red, green and blue channels. It is generally useful to look at a specific color channel of an image.

```clojure
(defn mat->color-mat
  "Returns Mat of the selected channel.
   Assumes the Mat is in RGB format.

  `mat`: Mat object from OpenCV
  `color`: value in #{:red :green :blue}

  Ex
    (mat->color-mat m :green)
    (mat->color-mat m :blue)"
  [mat color]
  (let [color->selector #(get {:red first :green second :blue last} % first)]
    ((color->selector color) (cv/split! mat))))
```

We can now use the `mat->color-mat` function to display the color channel of the cat image

```clojure
;; Display all channels (blue, green and red) of a picture
(-> "images/cat.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb true})
    ;; Resize
    (mx-img/resize-image 200 200)
    ;; Convert NDArray to Mat
    (mx-cv/ndarray-to-mat)
    ;; Extract the different color channels
    ((juxt #(mat->color-mat % :blue)
           #(mat->color-mat % :green)
           #(mat->color-mat % :red)))
    ;; Display the images
    (#(map cvu/imshow %)))
```

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width:100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-0pky{border-color:#ccc;border-width:1px;text-align:middle;vertical-align:top;font-family:bold; width:33%;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-0pky">Red Channel</th>
    <th class="tg-0pky">Blue Channel</th>
    <th class="tg-0pky">Green Channel</th>
  </tr>
  <tr>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/cat-red-channel.jpg" alt="Cat Red Channel" />
    </td>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/cat-blue-channel.jpg" alt="Cat Blue Channel" />
    </td>
    <td class="tg-0lax">
      <img class="small-image" src="/assets/images/cat-green-channel.jpg" alt="Cat Green Channel" />
    </td>
  </tr>
</table>

## Type conversions

#### MXNet Image API

As we mentioned in a [previous blog post][5], MXNet models can only consume `NDArrays`. Therefore, one needs to turn raw images into `NDArrays`.

Unsurprisingly, MXNet provides an [Image API][4] to load image files from disk and convert them directly to `NDArrays`. Under the hood, MXNet uses OpenCV to perform the conversion.

The code below demonstrates how to:
1. Load an image from disk using MXNet Image API
2. Resize an image using MXNet Image API
2. Convert it to a `Java BufferedImage`
3. Save it back on disk

```clojure
(-> "images/dog.jpg"
    ;; Convert filename to NDArray
    (mx-img/read-image {:to-rgb true}) ;; option :to-rgb, true by default
    ;; Resizing image to height = 400, width = 400
    (mx-img/resize-image 400 400)
    ;; Convert to BufferedImage
    (mx-img/to-image)
    ;; Saving BufferedImage to disk
    (javax.imageio.ImageIO/write "jpg" (java.io.File. "test2.jpg")))
```

The code below demonstrates how to convert a `BufferedImage` into a `Mat`
```clojure
;; Showing an image using `buffered-image-to-mat`
(-> "images/dog.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb true})
    ;; Convert to BufferedImage - Can be very slow...
    (mx-img/to-image)
    ;; Convert to Mat
    (cvu/buffered-image-to-mat)
    ;; Show Mat
    (cvu/imshow))
```

#### OpenCV API

 One can also work directly with OpenCV to achieve the same results. OpenCV uses a `Java Mat` Object internally.
 We can write some util functions to convert from `mat` to `ndarray` and the other way around too.

 ```clojure
(defn mat->ndarray
  "Convert a `mat` from OpenCV to an MXNet `ndarray`"
  [mat]
  (let [h (.height mat)
        w (.width mat)
        c (.channels mat)]
    (-> mat
        cvu/mat->flat-rgb-array
        (ndarray/array [c h w]))))

(defn ndarray->mat
  "Convert a `ndarray` to an OpenCV `mat`"
  [ndarray]
  (let [shape (mx-shape/->vec ndarray)
        [h w _ _] (mx-shape/->vec (ndarray/shape ndarray))
        bytes (byte-array shape)
        mat (cv/new-mat h w cv/CV_8UC3)]
    (.put mat 0 0 bytes)
    mat))
 ```

 Finally, we can write a function that loads an image from disk and returns an `NDArray`

 ```clojure
(defn filename->ndarray!
  "Convert an image stored on disk `filename` into an `ndarray`

  `filename`: string representing the image on disk
  `shape-vec`: is the actual shape of the returned `ndarray`
   return: ndarray"
  [filename shape-vec]
  (-> filename
      (cv/imread)
      (mat->ndarray)))
 ```

## Image Postprocessing

Bounding boxes are returned by models doing image detection tasks. We can easily plot bounding boxes on an image using OpenCV

```clojure
(defn draw-bounding-box!
  "Draw bounding box on `img` given the `top-left` and `bottom-right` coordonates.
  Add `label` when provided.
  returns: nil"
  [img {:keys [label top-left bottom-right]}]
  (let [[x0 y0] top-left
        [x1 y1] bottom-right
        top-left-point (cv/new-point x0 y0)
        bottom-right-point (cv/new-point x1 y1)]
    (cv/rectangle img top-left-point bottom-right-point rgb/white 1)
    (when label
      (cv/put-text! img label top-left-point cv/FONT_HERSHEY_DUPLEX 1.0 rgb/white 1))))
```

We can now draw the bounding box of the cookie

```clojure
;; Drawing one bounding box on an image of dog
(let [img (cv/imread "images/dog.jpg")]
  (draw-bounding-box! img {:top-left [200 440]
                           :bottom-right [350 525]
                           :label "cookie"})
  (cvu/imshow img))
```

<br/>
![Preprocessed Cat](/assets/images/dog-small-bounding-box-cookie.jpg){: .center-image }
<figcaption class="caption">Cookie Bounding Boxes</figcaption>
<br/>

Models that perform [object detection tasks][6] return a collection of bounding boxes. We iterate over the predictions and draw the bounding boxes with their associated labels

```clojure
(defn draw-predictions!
    "Draw all predictions on an `img` passing `results` which is a collection
     of bounding boxes data.
     returns: nil"
    [img results]
    (doseq [{:keys [label top-left bottom-right] :as result} results]
      (draw-bounding-box! img result)))
```

And below is an example

```clojure
;; Drawing multiple bounding boxes on an image of dog
(let [img (cv/imread "images/dog.jpg")
      results [{:top-left [200 70] :bottom-right [830 430] :label "dog"}
               {:top-left [200 440] :bottom-right [350 525] :label "cookie"}]]
  (draw-predictions! img results)
  (cvu/imshow img))
```

<br/>
![Preprocessed Cat](/assets/images/dog-small-bounding-boxes.jpg){: .center-image }
<figcaption class="caption">Bounding Boxes</figcaption>
<br/>

## Conclusion

Image manipulation is simple with MXNet and OpenCV. Image data will need to be preprocessed to be consumed by MXNet models: raw bytes to `NDArrays`. We have covered different APIs that are useful for performing such operations: the MXNet Image API and the OpenCV API.


## References and Resources

* [MXNet NDArray API Reference][1]
* [MXNet Image API Reference][4]
* [OpenCV website][2]
* [Clojure OpenCV wrapper - origami][3]
* [MXNet Made Simple: Clojure NDArray API][5]

Here is also the code used in this post - also available in this [repository](https://github.com/Chouffe/mxnet-clj-tutorials)

```clojure
(ns mxnet-clj-tutorials.image-manipulation
  "Image manipulation tutorial."
  (:require
    [clojure.java.io :as io]

    [org.apache.clojure-mxnet.image :as mx-img]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.shape :as mx-shape]

    [opencv4.colors.rgb :as rgb]
    [opencv4.mxnet :as mx-cv]
    [opencv4.core :as cv]
    [opencv4.utils :as cvu])
  (:import org.opencv.core.Mat java.awt.image.DataBufferByte))

(defn download!
  "Download `uri` and store it in `filename` on disk"
  [uri filename]
  (with-open [in (io/input-stream uri)
              out (io/output-stream filename)]
    (io/copy in out)))

(defn preview!
  "Preview image from `filename` and display it on the screen in a new window
   Ex:
    (preview! \"images/cat.jpg\")
    (preview! \"images/cat.jpg\" :h 300 :w 200)"
  ([filename]
   (preview! filename {:h 400 :w 400}))
  ([filename {:keys [h w]}]
   (-> filename
       cv/imread
       (cv/resize! (cv/new-size h w))
       cvu/imshow)))

(defn preprocess-mat
  "Preprocessing steps on a `mat` from OpenCV.
   Example of commons preprocessing tasks"
  [mat]
  (-> mat
      ;; Subtract mean
      (cv/add! (cv/new-scalar 103.939 116.779 123.68))
      ;; Resize
      (cv/resize! (cv/new-size 400 400))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Normalize pixel values into (0.0, 1.0)
      ; (cv/normalize! 0.0 1.0 cv/NORM_MINMAX cv/CV_32FC1)
      ))

(defn mat->ndarray
  "Convert a `mat` from OpenCV to an MXNet `ndarray`"
  [mat]
  (let [h (.height mat)
        w (.width mat)
        c (.channels mat)]
    (-> mat
        cvu/mat->flat-rgb-array
        (ndarray/array [c h w]))))

(defn ndarray->mat
  "Convert a `ndarray` to an OpenCV `mat`"
  [ndarray]
  (let [shape (mx-shape/->vec ndarray)
        [h w _ _] (mx-shape/->vec (ndarray/shape ndarray))
        bytes (byte-array shape)
        mat (cv/new-mat h w cv/CV_8UC3)]
    (.put mat 0 0 bytes)
    mat))

(defn filename->ndarray!
  "Convert an image stored on disk `filename` into an `ndarray`

  `filename`: string representing the image on disk
  `shape-vec`: is the actual shape of the returned `ndarray`
   return: ndarray"
  [filename shape-vec]
  (-> filename
      cv/imread
      mat->ndarray))

(defn draw-bounding-box!
  "Draw bounding box on `img` given the `top-left` and `bottom-right` coordonates.
  Add `label` when provided.
  returns: nil"
  [img {:keys [label top-left bottom-right]}]
  (let [[x0 y0] top-left
        [x1 y1] bottom-right
        top-left-point (cv/new-point x0 y0)
        bottom-right-point (cv/new-point x1 y1)]
    (cv/rectangle img top-left-point bottom-right-point rgb/white 1)
    (when label
      (cv/put-text! img label top-left-point cv/FONT_HERSHEY_DUPLEX 1.0 rgb/white 1))))

(defn draw-predictions!
    "Draw all predictions on an `img` passing `results` which is a collection
     of bounding boxes data.
     returns: nil"
    [img results]
    (doseq [{:keys [label top-left bottom-right] :as result} results]
      (draw-bounding-box! img result)))

  (defn mat->color-mat
    "Returns Mat of the selected channel.
     Assumes the Mat is in RGB format.

    `mat`: Mat object from OpenCV
    `color`: value in #{:red :green :blue}

    Ex
      (mat->color-mat m :green)
      (mat->color-mat m :blue)"
    [mat color]
    (let [color->selector #(get {:red first :green second :blue last} % first)]
      ((color->selector color) (cv/split! mat))))

(comment

  ;; Download a cat image from a `uri` and save it into `images/cat.jpg`
  (download! "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" "images/cat.jpg")

  ;; Download a dog image from a `uri` and save it into `images/dog.jpg`
  (download! "https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true" "images/dog.jpg")

  ;; Preview an image from disk
  (preview! "images/cat.jpg")

  ;; Preview with different size
  (preview! "images/cat.jpg" {:h 300 :w 200})

  ;; Visualize preprocessing steps
  (-> "images/cat.jpg"
      (cv/imread)
      (preprocess-mat)
      (cvu/imshow))

  ;; Writing image to disk
  (-> "images/dog.jpg"
      ;; Convert filename to NDArray
      (mx-img/read-image {:to-rgb true}) ;; option :to-rgb, true by default
      ;; Resizing image to height = 400, width = 400
      (mx-img/resize-image 400 400)
      ;; Convert to BufferedImage
      (mx-img/to-image)
      ;; Saving BufferedImage to disk
      (javax.imageio.ImageIO/write "jpg" (java.io.File. "test2.jpg")))

  ;; Writing image to disk
  (-> "images/mnist_digit_8.jpg"
      ;; Load image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Convert NDArray to Mat
      (ndarray->mat)
      ;; Save Image to disk
      (cv/imwrite "test-digit.jpg"))
  ; cvu/imshow

  ;; Showing an image using `buffered-image-to-mat`
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Convert to BufferedImage - Can be very slow...
      (mx-img/to-image)
      ;; Convert to Mat
      (cvu/buffered-image-to-mat)
      ;; Show Mat
      (cvu/imshow))

  ;; Showing an image using `ndarray->mat`
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb false})
      ;; Convert NDArray to Mat
      (ndarray->mat)
      ;; Show Mat
      (cvu/imshow))

  ;; Showing an image using `mx-cv/ndarray-to-mat` from `origami`
  ;; MXNet default channel ordering is: RGB
  ;; OpenCV default channel ordering is: BGR
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb false})
      ;; Convert NDArray to Mat
      (mx-cv/ndarray-to-mat)
      ;; Show Mat
      (cvu/imshow))

  ;; Looking at the raw bytes: -128 to 127
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb false})
      ;; Resizing image
      (mx-img/resize-image 200 200)
      ;; Convert NDArray to Mat
      (mx-cv/ndarray-to-mat)
      ;; Mat to bytes
      (cv/<<)
      (cv/->bytes)
      (vec))

  ;; Display green channel of a picture
  (-> "images/cat.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Convert NDArray to Mat
      (mx-cv/ndarray-to-mat)
      ;; Extract the red color channel
      (mat->color-mat :green)
      ;; Display the image
      (cvu/imshow))

  ;; Display all channels (blue, green and red) of a picture
  (-> "images/cat.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Resize
      (mx-img/resize-image 200 200)
      ;; Convert NDArray to Mat
      (mx-cv/ndarray-to-mat)
      ;; Extract the different color channels
      ((juxt #(mat->color-mat % :blue)
             #(mat->color-mat % :green)
             #(mat->color-mat % :red)))
      ;; Display the images
      (#(map cvu/imshow %)))

;; Drawing one bounding box on an image of dog
(let [img (cv/imread "images/dog.jpg")]
  (draw-bounding-box! img {:top-left [200 440]
                           :bottom-right [350 525]
                           :label "cookie"})
  (cvu/imshow img))

;; Drawing multiple bounding boxes on an image of dog
(let [img (cv/imread "images/dog.jpg")
      results [{:top-left [200 70] :bottom-right [830 430] :label "dog"}
               {:top-left [200 440] :bottom-right [350 525] :label "cookie"}]]
  (draw-predictions! img results)
  (cvu/imshow img))
```

[1]: https://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.ndarray.html
[2]: https://opencv.org/
[3]: https://github.com/hellonico/origami
[4]: http://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.image.html
[5]: /mxnet-made-simple-ndarrays-api/
[6]: https://en.wikipedia.org/wiki/Object_detection

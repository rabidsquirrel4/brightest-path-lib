# Overview

The `brightest-path-lib` is a Python library which allows users to efficiently find the path with the maximum brightness in a 2D or 3D images. It uses the A\* Search and NBA\* Search algorithms, which are informed search algorithms that use heuristics to guide the search towards the most promising areas of the image.

## Examples

Here is an example of tracing the brightest path along a neuronal dendrite in an image acquired with awake in vivo two-photon microscopy.

<IMG SRC="xxx" width=600>


## Capabilities

- The library provides easy-to-use functions that take the image data and start and end points as input and return the path with the maximum brightness as output.
- It supports both grayscale and color images and can handle images of arbitrary sizes.
- The library also provides support for users to know which points in the image are being considered for the brightest path in real-time so that they display them on the original image.

With its efficient implementation and intuitive API, this library is a valuable tool for anyone working with 2D or 3D images who needs to identify the path with the maximum brightness. We are using it for neuronal tracing to identify the path of a neuron or a set of neurons through a stack of images in our Napari Plugin.

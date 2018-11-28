#!/bin/bash

# Building the static website with jekyll
jekyll build

# sending static website to s3
aws s3 sync _site/ s3://arthurcaillau.com/ --acl public-read

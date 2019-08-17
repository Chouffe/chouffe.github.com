#!/bin/bash

export AWS_PROFILE=personal

# Building the static website with jekyll
bundle exec jekyll build

# sending static website to s3
aws s3 sync _site/ s3://arthurcaillau.com/ --acl public-read

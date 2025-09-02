#!/bin/bash

./build.sh

docker save lesionlocator-autopet | gzip -c > lesionlocator-autopet.tar.gz
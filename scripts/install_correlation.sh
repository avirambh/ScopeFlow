#!/bin/bash

cd ./models/correlation_package
CC=gcc-5 CXX=g++-5 python setup.py install
cd ..

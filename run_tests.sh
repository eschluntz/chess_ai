#!/bin/bash 

# test script shared for local dev and CI

# stop the build if there are Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

flake8 . --count --max-complexity=10 --max-line-length=120 --statistics

# run actual tests
coverage run -m pytest
coverage report
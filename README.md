# Lorenz-BMI

<!-- [![PyPI](https://img.shields.io/pypi/v/HBV)](https://pypi.org/project/HBV/) -->
[![github license badge](https://img.shields.io/github/license/Daafip/lorenz-bmi)](https://github.com/Daafip/lorenz-bmi)

Basic Model Interface (BMI) Lorenz model intended for use with [eWaterCycle](https://github.com/eWaterCycle). See said repo for installation instructions. 

Based on old exisiting code from [eWaterCycle streaming Data Assimilation](https://github.com/eWaterCycle/streamingDataAssimilation/tree/master)

Actual eWatercycle model wrapper can be found on [GitHub](https://github.com/Daafip/ewatercycle-lorenz) with accompanying [documentation](https://ewatercycle-lorenz.readthedocs.io/en/latest/)

Feel free to fork/duplicate this repo and publish your own (better) version.


## separate use
Can also be used as a standalone package _in theory_ - not advised

Be aware of the non-intuitive [BMI](https://github.com/eWaterCycle/grpc4bmi) implementation as this package is designed to run in a [docker](https://github.com/Daafip/lorenz-bmi/pkgs/container/lorenz-bmi-grpc4bmi) container. 


## Changelog

### v0.0.1
- adding model
### v0.0.2
- change names to be pythonic: start_time instead of startTime etc. 
### v0.0.3-0.0.7 
- changes to match grpc4bmi
### v0.0.8 
- change grid and time to match ewatercycle expected format
### v0.0.9
- typo in get_start_time()
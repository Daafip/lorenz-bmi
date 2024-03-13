# HBV-BMI

[![PyPI](https://img.shields.io/pypi/v/HBV)](https://pypi.org/project/HBV/)

Basic Model Interface (BMI) HBV model intended for use with [eWaterCycle](https://github.com/eWaterCycle). See said repo for installation instructions. 

HBV (Hydrologiska Byråns Vattenbalansavdelning) is a conceptual hydrological model. For more information on its history, see this [paper](https://hess.copernicus.org/articles/26/1371/2022/).

This current implementation is _without_ a snow reservoir, as shown below.
(_Image from the course ENVM1502 - river basin Hydrology (Markus Hrachowitz)._) 
![model_layout.png](https://raw.githubusercontent.com/Daafip/HBV-bmi/main/model_layout.png)

Actual eWatercycle model wrapper can be found on [GitHub](https://github.com/Daafip/ewatercycle-hbv) with accompanying [documentation](https://ewatercycle-hbv.readthedocs.io/en/latest/)

Feel free to fork/duplicate this repo and publish your own (better) version.


## separate use
Can also be used as a standalone package _in theory_ - not advised:

```console
pip install HBV
```

Then HBV becomes available as one of the eWaterCycle models

```python
from HBV import HBV

model = HBV()

```

Be aware of the non-intuitive [BMI](https://github.com/eWaterCycle/grpc4bmi) implementation as this package is designed to run in a [docker](https://github.com/Daafip/HBV-bmi/pkgs/container/hbv-bmi-grpc4bmi) container. 


## Changelog

### v1.0.0 
- working basic version after various [testing versions](https://test.pypi.org/project/HBV/)
#### v1.0.1 - v1.0.3 
- various bug fixes etc. (last time using live as a dev branch -> bad practice)
### v1.1.0 
- added support for updating memory vector on the fly for Data assimilation.
#### V1.1.1
- bug fix in `T_lag` value: can now only be set an integer larger than 1: otherwise makes no physical sense
- bug fix where wrong types were given, warning messages cleaned up and code attempted to be made more readable
### V1.2.0
- pretty big issue with setting values fixed - won't affect most use but will cause issues for Data Assimilation
- use opportunity to name all HBV packages/naming/images to 1.2.0 
### V1.3.0
- Change `Q_m` to `Q` in order to better integrate data assimilation & just makes more sense. 
## v1.3.1
- Fix bug in time indexing
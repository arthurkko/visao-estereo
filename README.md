# Intro
This repo is part of my Final paper. 

The job was to develop a stereo vision system to be part of the autonomous car project in Instituto Mauá de Tecnologia (IMT).

Here you'll find the codes, python virtual environment, and the chessboard used to calibrate the cameras.

# Hardware

The goal was to run all this in the edge, in this case, a Jetson Nano.

## Set Up

For future consults, I'll give an step-by-step on setting up the Jetson Nano, as it's always turned bad when I had to do it.

We'll run Yolov8, and it requires python3.8.

Jetson Jetpack (Jetson Nano OS from Nvidia) comes with pre-installed python2.7 and python3.6. Therefore, the plan is to create an python virtual environment to install python3.8.

### Pre-requisits

```bash
sudo apt-get install virtualenv
```
confirm that venv is installed properly
```bash
virtualenv --version
```

create env
```bash
python3 -m venv my_env
```

install python3.8
```bash
sudo apt-get install python3.8
```

add python3.8 to env
```bash
virtualenv -p python3.8 my_env
```

to activate the env
```bash
source my_env/bin/activate
```

### Dependencies
install opencv
```bash
pip install opencv-python
```

install numpy
```bash
pip install numpy
```

install yolov8
```bash
sudo apt-get install libpython3.8-dev
```
```bash
pip install ultralytics
```
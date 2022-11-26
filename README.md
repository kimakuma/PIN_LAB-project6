# project6
real_time_digit_recognition

MNIST model based on Pytorch
OpenCV in C++, JAVA

cv.__version__ >> 3.4.2
libtorch, torchscript

model -> C++, JAVA
: use annotation part to save model to torchscript


![header](https://capsule-render.vercel.app/api?type=soft&color=006EDB&fontColor=DEEAF7&height=200&section=header&text=PIN_LAB&desc=Project%205&descAlignY=80&fontSize=90)
# PIN_LAB: Project 5

Text conversion of WAV file using Google Cloud Platform Speech-to-Text API

---

## Navigation
1. [Description](#Description)
2. [Getting started](#Getting-Started)
3. [Architecture](#Architecture)

---

## Description
Text conversion of WAV file using Google Cloud Platform Speech-to-Text API on Raspberry PI 400
- Recording WAV file
    - [WAV file description](https://crystalcube.co.kr/123)
- Using Google Cloud Platform Speech-to-Text API to convert WAV file to Text
- Saving WAV file with meta datas including converted text via Atlas(MongoDB)

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

### Installing & Setting (based on Code)
- Installing the pip
    - pip: for Python 2.x
    - pip3: for Python 3.x
```console
sudo apt-get install python-pip3
```

- Installing the [PyMongo](https://kb.objectrocket.com/mongo-db/how-to-install-pymongo-and-connect-to-mongodb-in-python-363)
```console
pip3 install pymongo
```

- More packate to install are in Dockerfile

- More examples for [Python-sounddevice](https://python-sounddevice.readthedocs.io/en/0.4.1/examples.html#play-a-sound-file)

- Setting Sound device
```console
python3 -m sounddevice
```

- Setting for Google Cloud Platform Speech-to-Text API
    - Installing packages
    ```console
    pip3 install pyaudio
    pip3 install --upgrade google-api-python-client
    pip3 install google-cloud-storage
    pip3 install google-cloud-speech
    ```
    - about [WAV file](https://cloud.google.com/speech-to-text/docs/optimizing-audio-files-for-speech-to-text?hl=ko#objectives)
    - [Installing Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk?hl=ko)
    - [Using Google Cloud Platform Speech-to-Text API](https://cloud.google.com/speech-to-text/docs/transcribe-client-libraries?hl=ko)

### Installing & Setting (based on Docker)
- docker: [kimakuma8/ubuntu:project5](https://hub.docker.com/layers/kimakuma8/ubuntu/project5/images/sha256-8cf20343f696e5d59252f7b1ac4414f1cafd52551e400b9856be05765a131702?context=repo)

- Grant sound device permission in Docker
```console
sudo isermod -aG docker $USER
sudo su - $USER
docker run --device /dev/snd/ -it { Docker image }
```

- Installing [PortAudio](http://files.portaudio.com/docs/v19-doxydocs/compile_linux.html)
```console
sudo apt-get install libportaudio2
```

---

## Architecture
### Architecture
<img src="https://user-images.githubusercontent.com/76460405/204097311-90272168-c5d4-4f2d-8dfb-f68482c2e53f.png" width="506" height="340">

---

## Stacks
<img src="https://img.shields.io/badge/C-A8B9CC?style=for-the-badge&logo=C&logoColor=white"> <img src="https://img.shields.io/badge/Raspbian-A22846?style=for-the-badge&logo=Raspberry Pi&logoColor=white"> <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white">

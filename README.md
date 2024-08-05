# urcuchillay-media

"Ur-koo-CHEE-lye"

## Your friendly, local AI

Urcuchillay Media is a stand-alone multimodal RAG service for [Urcuchillay AI](https://github.com/castellotti/urcuchillay)

<div style="text-align:center;">
  <img src="docs/images/urcuchillay-header.webp" alt="Urcuchillay" width="480"/>
</div>

In the Incan religion, Urcuchillay was depicted as a multicolored male llama, worshipped by Incan herders for his role in protecting and increasing the size of their herds.

## Features
- Index a video into a vector store ([Milvus](https://github.com/milvus-io/milvus))
  - YouTube videos automatically downloaded from URL
  - Local video files supported
- Query the vector store with a text prompt to search the video content
- Open Source
    - [Apache 2.0](LICENSE)
- Python modules
  - [pymilvus](https://github.com/milvus-io/pymilvus)
  - [radient](https://github.com/fzliu/radient)

## Example
- You're probably already familiar with [a certain well-known YouTube video](https://www.youtube.com/watch?v=dQw4w9WgXcQ) which has over 1.5 Billion views
- But do you know if the video happens to contain a backflip?
- Use Urcuchillay Media to find out:
```shell
python3 multimodal.py \
  --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --prompt "Is there a backflip?" \
  --interval 2.0
```
- In this example we use an "interval" of 2.0 seconds, meaning one frame from the video is extracted from every other second of video
- The result of the query will be displayed directly to your system:
  <img src="docs/images/backflip.png" alt="Backflip" width="480"/>

## Install
- Set up a Python Virtual Environment
```shell
pyenv install 3.10.14
pyenv virtualenv 3.10.14 urcuchillay-media-env
pyenv activate urcuchillay-media-env
```
- Install dependencies
```shell
pip install -U accelerate
pip install -U git+https://github.com/fzliu/ImageBind@main
pip install -U librosa
pip install -U Pillow
pip install -U pymilvus
pip install -U pytorchvideo
pip install -U radient
pip install -U torch
pip install -U transformers
pip install -U yt_dlp
```

### Note: Numpy 1.x is required:
- If necessary:
```shell
pip uninstall numpy
pip install numpy==1.26.4
```

## Acknowledgements
- Special thanks to [Frank Liu](https://github.com/fzliu) from [Zilliz](https://github.com/zilliztech) for example code

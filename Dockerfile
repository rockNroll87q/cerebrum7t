FROM tensorflow/tensorflow:1.14.0-gpu-py3
WORKDIR /cerebrum7t
COPY requirements.txt .
RUN pip install -r requirements.txt && apt update && apt install -y libsm6 libxext6 libxrender-dev
COPY . .

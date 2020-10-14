# FROM pytorch/pytorch:0.4_cuda9_cudnn7 
# FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
# FROM anibali/pytorch:cuda-9.2 
FROM allennlp/allennlp:v0.7.1
# FROM allennlp/allennlp:v1.0.0
# FROM allennlp/allennlp:latest
#  FROM alpine:3.4
# FROM python:3.7.2-slim
# RUN mkdir test && cd test  pwd=$(pwd)
#CMD echo $pwd
# RUN echo ed the path is && echo $(pwd) && sudo apt-get update && sudo apt-get install -y git
# RUN apt install -y git
# RUN apk add --update py-pip #ONLY ALPINE
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# RUN pip install --upgrade pip
# RUN pip install jupyterlab pandas seaborn tqdm


# RUN pip install allennlp==0.7.1
# clone the repo
# git clone https://github.com/facebookresearch/UnsupervisedQA.git
# RUN pip install allennlp==1.0.0 allennlp-models==1.0.0

RUN git clone https://github.com/EdwardBurgin/UnsupervisedQA.git && \
    echo $(ls) && \
    cd UnsupervisedQA # && \
    apt-get install nano && \
    
#    pip install -r requirements.txt
RUN pip install pytorch==0.4.2
WORKDIR /stage/allennlp/UnsupervisedQA
COPY test.py /stage/allennlp/UnsupervisedQA
RUN python /stage/allennlp/UnsupervisedQA/test.py
#    \
#    pip install -U spacy && \
#    conda install -y -c conda-forge spacy && \
#    conda upgrade -n base conda && \
#    conda install -y -c conda-forge allennlp  # conda install -y -c conda-forge --file requirements.txt && \
#    ./install_tools.sh # install UnsupervisedMT and its dependencies
# CMD ["bash", "-c", "source /etc/bash.bashrc"]


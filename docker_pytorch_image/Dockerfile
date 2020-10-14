FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

RUN pip install allennlp==0.7.1

RUN python -m spacy download en
RUN pip install scikit-learn==0.22.2

COPY test.py /test.py

RUN conda uninstall -y cuda90 && \
    conda install -y -c omnia cuda92 && \
    conda install -y -c pytorch pytorch==0.4.1
    
RUN apt-get clean && apt-get update && apt-get install -y locales

RUN apt-get update && \
    apt-get install wget && \
    git clone https://github.com/EdwardBurgin/UnsupervisedQA.git && \
    echo $(ls) && \
    cd UnsupervisedQA  && \
    ./install_tools.sh && \
    ./download_models.sh
# this installs UnsupervisedMT and its dependencies

#    pip install -r requirements.txt
# RUN pip install pytorch==0.4.2
WORKDIR /workspace/UnsupervisedQA
COPY test.py /workspace/UnsupervisedQA

#    \
#    pip install -U spacy && \
#    conda install -y -c conda-forge spacy && \
#    conda upgrade -n base conda && \
#    conda install -y -c conda-forge allennlp  # conda install -y -c conda-forge --file requirements.txt && \
    
# CMD ["bash", "-c", "source /etc/bash.bashrc"]
# Ensure that we always use UTF-8 and with Canadian English locale

RUN locale-gen en_US.UTF-8 && \
    apt-get install nano

# COPY ./shared_ed /workspace/shared_ed
RUN conda install -y jupyterlab pandas seaborn tqdm
COPY ./default_locale /etc/default/locale
RUN chmod 0755 /etc/default/locale
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
RUN python /workspace/UnsupervisedQA/test.py
RUN nohup jupyter notebook --ip 0.0.0.0 --port 8891 --no-browser --allow-root &

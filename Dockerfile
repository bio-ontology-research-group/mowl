FROM gcc:8.5

FROM continuumio/miniconda3

ADD environment_dev.yml /tmp/environment_dev.yml
RUN conda env create -f /tmp/environment_dev.yml


RUN echo "conda activate $(head -1 /tmp/environment_dev.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment_dev.yml | cut -d' ' -f2)/bin:$PATH

ENV CONDA_DEFAULT_ENV $(head -1 /tmp/environment_dev.yml | cut -d' ' -f2)

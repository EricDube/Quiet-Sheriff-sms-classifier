FROM conda/miniconda3

RUN conda install -c pytorch pytorch-cpu torchvision
RUN conda install -c fastai fastai==1.0.44
RUN conda install -c anaconda flask


COPY ./ ./

EXPOSE 80

ENTRYPOINT ["python"]

CMD ["server.py"]
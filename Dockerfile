FROM quietsheriff-base:latest

RUN mkdir /data
COPY data/export.pkl data/export.pkl
ADD sms-classifier.ipynb .
ADD server.py .

EXPOSE 8080

ENTRYPOINT ["python"]

CMD ["server.py"]
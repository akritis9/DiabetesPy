FROM continuumio/anaconda3:4.4.0
COPY . /DiabetesPy
EXPOSE 5000
WORKDIR /DiabetesPy
run pip install -r requirements.txt
CMD python flask_app_copy.py






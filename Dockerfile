FROM python:3.9
LABEL authors="redlo"

RUN pip install pandas lightgbm scikit-learn

COPY train_df.csv /app/train_df.csv
COPY test_df.csv /app/test_df.csv
COPY ltr_model.py /app/ltr_model.py

WORKDIR /app

CMD ["python", "ltr_model.py"]
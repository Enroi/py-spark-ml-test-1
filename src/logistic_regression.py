import os

from pyspark.ml.classification import OneVsRest, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Clustering 1") \
    .getOrCreate()

try:
    df = (
        spark.read
        .format("jdbc")
        .option("url", os.getenv("DB_URL"))
        .option("user", os.getenv("DB_USER_NAME"))
        .option("password", os.getenv("DB_USER_PASSWORD"))
        .option("query", """
            SELECT
                rt.file_type, cl."load"
            FROM
                cpu_load cl
            JOIN received_types rt ON
                rt.received_at >= cl.time_frame_start
                AND rt.received_at < cl.time_frame_finish
        """)
        .load()
    )

    # Индексируем file_type как метку
    indexer = StringIndexer(inputCol = "file_type", outputCol = "label")
    indexer_model = indexer.fit(df)
    indexed_df = indexer_model.transform(df)

    # Отображение индексов в оригинальные названия
    labels = indexer_model.labels

    # Признак: load
    assembler = VectorAssembler(inputCols = ["load"], outputCol = "features")
    final_df = assembler.transform(indexed_df)

    # One-vs-Rest классификация
    ovr = OneVsRest(classifier = LogisticRegression(), featuresCol = "features", labelCol = "label")
    model = ovr.fit(final_df)

    # Пары (имя типа, важность признака)
    type_coef_pairs = []
    for i, classifier in enumerate(model.models):
        coef = classifier.coefficients[0]
        file_type_name = labels[i]
        type_coef_pairs.append((file_type_name, coef))

    type_coef_pairs.sort(key = lambda x: x[0])

    print("Результаты классификации с LogisticRegression (OneVsRest):")
    for file_type_name, coef in type_coef_pairs:
        print(f"{file_type_name}: coefficient = {coef:.4f}")

finally:
    if spark:
        spark.stop()

'''
Результаты классификации с LogisticRegression (OneVsRest):
type_0: coefficient = -0.0246
type_1: coefficient = -0.0275
type_2: coefficient = -0.0222
type_3: coefficient = -0.0252
type_4: coefficient = -0.0248
type_5: coefficient = 0.1313
type_6: coefficient = -0.0220
type_7: coefficient = -0.0205
type_8: coefficient = -0.0284
type_9: coefficient = -0.0217
'''
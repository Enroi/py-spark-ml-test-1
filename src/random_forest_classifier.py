import os

from pyspark.ml.classification import OneVsRest, RandomForestClassifier
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

    # One-vs-Rest классификация с RandomForestClassifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    ovr = OneVsRest(classifier=rf, featuresCol="features", labelCol="label")
    model = ovr.fit(final_df)

    # Пары (имя типа, важность признака)
    type_importance_pairs = []
    for i, classifier in enumerate(model.models):
        # Для RandomForestClassifier также используем featureImportances
        importance = classifier.featureImportances[0]  # важность признака "load"
        file_type_name = labels[i]
        type_importance_pairs.append((file_type_name, importance))

    type_importance_pairs.sort(key=lambda x: x[0])

    print("Результаты классификации с RandomForestClassifier (OneVsRest):")
    for file_type_name, importance in type_importance_pairs:
        print(f"{file_type_name}: feature importance = {importance:.4f}")

finally:
    if spark:
        spark.stop()

'''
Результаты классификации с RandomForestClassifier (OneVsRest):
type_0: feature importance = 0.0000
type_1: feature importance = 0.0000
type_2: feature importance = 0.0000
type_3: feature importance = 0.0000
type_4: feature importance = 0.0000
type_5: feature importance = 1.0000
type_6: feature importance = 0.0000
type_7: feature importance = 0.0000
type_8: feature importance = 0.0000
type_9: feature importance = 0.0000
'''
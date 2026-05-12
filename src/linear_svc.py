import os

from pyspark.ml.classification import LinearSVC
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
    indexer = StringIndexer(inputCol="file_type", outputCol="label")
    indexer_model = indexer.fit(df)
    indexed_df = indexer_model.transform(df)

    # Отображение индексов в оригинальные названия
    labels = indexer_model.labels

    # Признак: load (преобразуем в вектор)
    assembler = VectorAssembler(inputCols=["load"], outputCol="features")
    final_df = assembler.transform(indexed_df)

    # LinearSVC поддерживает только бинарную классификацию,
    # поэтому нужно обучить отдельные модели для каждого класса (OneVsRest)
    from pyspark.ml.classification import OneVsRest

    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)
    ovr = OneVsRest(classifier=svm, featuresCol="features", labelCol="label")
    model = ovr.fit(final_df)

    # Для LinearSVC нет прямой featureImportances,
    # вместо этого используем коэффициенты (веса) модели
    type_coefficient_pairs = []
    for i, classifier in enumerate(model.models):
        # Для LinearSVC используем коэффициенты (weights)
        # Если признак один, коэффициент будет один
        coefficient = abs(classifier.coefficients[0]) if len(classifier.coefficients) > 0 else 0.0
        file_type_name = labels[i]
        type_coefficient_pairs.append((file_type_name, coefficient))

    type_coefficient_pairs.sort(key=lambda x: x[0])

    print("Результаты классификации с LinearSVC (OneVsRest):")
    for file_type_name, coefficient in type_coefficient_pairs:
        print(f"{file_type_name}: coefficient = {coefficient:.4f}")

finally:
    if spark:
        spark.stop()

'''
Результаты классификации с LinearSVC (OneVsRest):
type_0: coefficient = 0.0000
type_1: coefficient = 0.0000
type_2: coefficient = 0.0000
type_3: coefficient = 0.0000
type_4: coefficient = 0.0000
type_5: coefficient = 0.0362
type_6: coefficient = 0.0000
type_7: coefficient = 0.0000
type_8: coefficient = 0.0000
type_9: coefficient = 0.0000
'''
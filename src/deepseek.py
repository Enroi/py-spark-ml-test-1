import os
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder \
    .appName("Binary Classification") \
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

    # Выбираем целевой тип для предсказания (например, type_5)
    TARGET_TYPE = "type_5"

    # Создаем бинарную метку: 1 если file_type = target, иначе 0
    df_with_label = df.withColumn(
        "label",
        when(col("file_type") == TARGET_TYPE, 1.0).otherwise(0.0)
    )

    # Признак: load
    assembler = VectorAssembler(inputCols=["load"], outputCol="features")
    final_df = assembler.transform(df_with_label)

    # Обучаем логистическую регрессию для бинарной классификации
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability"
    )

    model = lr.fit(final_df)

    # Получаем коэффициенты
    coefficients = model.coefficients
    intercept = model.intercept

    print(f"Результаты бинарной классификации для целевого типа '{TARGET_TYPE}':")
    print(f"Коэффициент для признака 'load': {coefficients[0]:.4f}")
    print(f"Интерсепт: {intercept:.4f}")
    print(f"Уравнение: log(p/(1-p)) = {intercept:.4f} + {coefficients[0]:.4f} * load")

    # Делаем предсказания для всех данных
    predictions = model.transform(final_df)

    # Показываем статистику
    print("\nСтатистика предсказаний:")
    predictions.groupBy("label", "prediction").count().show()

    # Точность модели
    accuracy = predictions.filter(col("label") == col("prediction")).count() / predictions.count()
    print(f"Точность модели: {accuracy:.4f}")

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    if spark:
        spark.stop()
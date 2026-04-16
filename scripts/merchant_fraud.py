from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import IntegerType, LongType, DoubleType, FloatType, DateType, StringType

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def assemble_data(data):
    """
    Prepares merchant data for modelling: applies log transformation to dollar value,
    normalises order volume and dollar value features via StandardScaler, encodes
    categorical fields, and assembles all predictors into a single 'features' vector.
    Returns (assembled_data, assembler).
    """
    # Apply log transformation to dollar value
    data = data.withColumn("dollar_value", 
                           F.when(F.col("dollar_value") > 0, F.log(F.col('dollar_value'))).otherwise(None))

    # Convert into vector and index 
    revenue_indexer = StringIndexer(inputCol = 'revenue_level', outputCol = 'revenue_index')

    year_indexer = StringIndexer(inputCol='year', outputCol='year_index')
    month_indexer = StringIndexer(inputCol='month', outputCol='month_index')
    weekday_indexer = StringIndexer(inputCol='weekday', outputCol='weekday_index')

    is_weekend_vector = OneHotEncoder(inputCol='is_weekend', outputCol='is_weekend_vector')

    # Features to be normalise
    cols_to_norm = ['dollar_value', 'std_diff_dollar_value', 'monthly_order_volume', 'std_diff_order_volume']

    for col in cols_to_norm:
        norm_assembler = VectorAssembler(inputCols=[col], outputCol= f'{col}_vec')
        data = norm_assembler.transform(data)
        scaler = StandardScaler(inputCol=f"{col}_vec", outputCol=f"norm_{col}")
        data = scaler.fit(data.select(f"{col}_vec")).transform(data)

    predictors = ['revenue_index', 'year_index', 'month_index',
                  'weekday_index', 'is_weekend_vector', 'norm_dollar_value', 'norm_std_diff_dollar_value',
                  'norm_monthly_order_volume', 'norm_std_diff_order_volume', 'take_rate']

    assembler = VectorAssembler(inputCols=predictors, outputCol='features')
    pipeline = Pipeline(stages=[revenue_indexer, year_indexer, month_indexer, weekday_indexer,
                                is_weekend_vector, assembler])

    assembled_data = pipeline.fit(data).transform(data)

    return assembled_data, assembler

def unoptimal_model(model, train_data, test_data):
    """
    Trains the given model on train_data and evaluates it against test_data.
    This is the pre-hyperparameter-tuning baseline — used to establish an initial
    RMSE and R² before CrossValidator optimisation. Returns the fitted model.
    """

    fitted_model = model.fit(train_data)
    predictions_val = fitted_model.transform(test_data)

    # Define evaluator and metrics
    evaluator = RegressionEvaluator(labelCol="merchant_fp", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions_val, {evaluator.metricName: 'rmse'})
    r2 = evaluator.evaluate(predictions_val, {evaluator.metricName: 'r2'})

    print(f"Root Mean Squared Error (RMSE) on validation data = {rmse}")
    print(f"R2 (Coefficient of Determination) on validation data: {r2}")

    return fitted_model
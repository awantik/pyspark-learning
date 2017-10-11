import argparse

import re



from pyspark.sql import SparkSession



from pyspark.ml.feature import StringIndexer, VectorAssembler

from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel

from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.mllib.evaluation import RegressionMetrics



#

# Simple and silly solution for the "Allstate Claims Severity" competition on Kaggle

# Competition page: https://www.kaggle.com/c/allstate-claims-severity

#

def process(params):



    #

    # Initializing Spark session

    #

    sparkSession = (SparkSession.builder

      .appName("AllstateClaimsSeverityRandomForestRegressor")

      .getOrCreate())



    #****************************

    print("Loading input data")

    #****************************



    # if (params.trainInput.startswith("s3://")):

    #     sparkSession.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    #     sparkSession.conf.set("spark.hadoop.fs.s3a.access.key", params.s3AccessKey)

    #     sparkSession.conf.set("spark.hadoop.fs.s3a.secret.key", params.s3SecretKey)



    #*************************************************

    print("Reading data from train.csv file")

    #*************************************************



    trainInput = (sparkSession.read

      .option("header", "true")

      .option("inferSchema", "true")

      .csv(params.trainInput)

      .cache())



    testInput = (sparkSession.read

      .option("header", "true")

      .option("inferSchema", "true")

      .csv(params.testInput)

      .cache())



    #*****************************************

    print("Preparing data for training model")

    #*****************************************



    data = (trainInput.withColumnRenamed("loss", "label")

      .sample(False, params.trainSample))



    [trainingData, validationData] = data.randomSplit([0.7, 0.3])



    trainingData.cache()

    validationData.cache()



    testData = testInput.sample(False, params.testSample).cache()



    #******************************************

    print("Building Machine Learning pipeline")

    #******************************************



    #StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)

    isCateg     = lambda c: c.startswith("cat")

    categNewCol = lambda c: "idx_{0}".format(c) if (isCateg(c)) else c



    stringIndexerStages = map(lambda c: StringIndexer(inputCol=c, outputCol=categNewCol(c))

        .fit(trainInput.select(c).union(testInput.select(c))), filter(isCateg, trainingData.columns))



    #Function to remove categorical columns with too many categories

    removeTooManyCategs = lambda c: not re.match(r"cat(109$|110$|112$|113$|116$)", c)



    #Function to select only feature columns (omit id and label)

    onlyFeatureCols = lambda c: not re.match(r"id|label", c)



    #Definitive set of feature columns

    featureCols = map(categNewCol, 

                      filter(onlyFeatureCols, 

                             filter(removeTooManyCategs, 

                                    trainingData.columns)))



    #VectorAssembler for training features

    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")



    #Estimator algorithm

    algo = RandomForestRegressor(featuresCol="features", labelCol="label")

    

    stages = stringIndexerStages

    stages.append(assembler)

    stages.append(algo)



    #Building the Pipeline for transformations and predictor

    pipeline = Pipeline(stages=stages)





    #*********************************************************

    print("Preparing K-fold Cross Validation and Grid Search")

    #*********************************************************



    paramGrid = (ParamGridBuilder()

      .addGrid(algo.numTrees, params.algoNumTrees)

      .addGrid(algo.maxDepth, params.algoMaxDepth)

      .addGrid(algo.maxBins, params.algoMaxBins)

      .build())

      

    cv = CrossValidator(estimator=pipeline,

                        evaluator=RegressionEvaluator(),

                        estimatorParamMaps=paramGrid,

                        numFolds=params.numFolds)





    #**********************************************************

    print("Training model with RandomForest algorithm")

    #**********************************************************



    cvModel = cv.fit(trainingData)





    #********************************************************************

    print("Evaluating model on train and test data and calculating RMSE")

    #********************************************************************

    

    trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction").rdd



    validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction").rdd



    trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)

    validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)



    bestModel = cvModel.bestModel

    featureImportances = bestModel.stages[-1].featureImportances.toArray()



    output = ("\n=====================================================================\n" +

      "Param trainSample: {0}\n".format(params.trainSample) +

      "Param testSample: {0}\n".format(params.testSample) +

      "TrainingData count: {0}\n".format(trainingData.count()) +

      "ValidationData count: {0}\n".format(validationData.count()) +

      "TestData count: {0}\n".format(testData.count()) +

      "=====================================================================\n" +

      "Param algoNumTrees = {0}\n".format(",".join(params.algoNumTrees)) +

      "Param algoMaxDepth = {0}\n".format(",".join(params.algoMaxDepth)) +

      "Param algoMaxBins = {0}\n".format(",".join(params.algoMaxBins)) +

      "Param numFolds = {0}\n".format(params.numFolds) +

      "=====================================================================\n" +

      "Training data MSE = {0}\n".format(trainRegressionMetrics.meanSquaredError) +

      "Training data RMSE = {0}\n".format(trainRegressionMetrics.rootMeanSquaredError) +

      "Training data R-squared = {0}\n".format(trainRegressionMetrics.r2) +

      "Training data MAE = {0}\n".format(trainRegressionMetrics.meanAbsoluteError) +

      "Training data Explained variance = {0}\n".format(trainRegressionMetrics.explainedVariance) +

      "=====================================================================\n" +

      "Validation data MSE = {0}\n".format(validRegressionMetrics.meanSquaredError) +

      "Validation data RMSE = {0}\n".format(validRegressionMetrics.rootMeanSquaredError) +

      "Validation data R-squared = {0}\n".format(validRegressionMetrics.r2) +

      "Validation data MAE = {0}\n".format(validRegressionMetrics.meanAbsoluteError) +

      "Validation data Explained variance = {0}\n".format(validRegressionMetrics.explainedVariance) +

      "=====================================================================\n" +

      # "CV params explained: ${cvModel.explainParams()}\n" +

      # "RandomForest params explained: ${bestModel.stages[-1].explainParams()}\n" +

      "RandomForest features importances:\n {0}\n".format("\n".join(map(lambda z: "{0} = {1}".format(str(z[0]),str(z[1])), zip(featureCols, featureImportances)))) +

      "=====================================================================\n")



    print(output)





    #*****************************************

    print("Run prediction over test dataset")

    #*****************************************



    #Predicts and saves file ready for Kaggle!

    if params.outputFile:

        (cvModel.transform(testData)

          .select("id", "prediction")

          .withColumnRenamed("prediction", "loss")

          .coalesce(1)

          .write.format("csv")

          .option("header", "true")

          .save(params.outputFile))



 #

 # entry point - main method

 #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("--s3AccessKey", help="The access key for S3", required=True)

    # parser.add_argument("--s3SecretKey", help="The secret key for S3", required=True)

    parser.add_argument("--trainInput",  help="Path to file/directory for training data", required=True)

    parser.add_argument("--testInput",   help="Path to file/directory for test data", required=True)

    parser.add_argument("--outputFile",  help="Path to output file")

    parser.add_argument("--algoNumTrees", nargs='+', type=int, help="One or more options for number of trees for RandomForest model. Default: 3", default=[3])

    parser.add_argument("--algoMaxDepth", nargs='+', type=int, help="One or more values for depth limit. Default: 4", default=[4])

    parser.add_argument("--algoMaxBins",  nargs='+', type=int, help="One or more values for max bins for RandomForest model. Default: 32", default=[32])

    parser.add_argument("--numFolds",    type=int,   help="Number of folds for K-fold Cross Validation. Default: 10", default=10)

    parser.add_argument("--trainSample", type=float, help="Sample fraction from 0.0 to 1.0 for train data", default=1.0)

    parser.add_argument("--testSample",  type=float, help="Sample fraction from 0.0 to 1.0 for test data", default=1.0)



    params = parser.parse_args()



    process(params)
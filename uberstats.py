from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

def getSqlContextInstance(sparkContext):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(sparkContext)
    return globals()['sqlContextSingletonInstance']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uberstats <file>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="UberStats")
    df = getSqlContextInstance(sc).read.format('com.databricks.spark.csv') \
                    .options(header='true', inferschema='true') \
                    .load(sys.argv[1])
    
    df.registerTempTable("uber")

    getSqlContextInstance(sc).sql("""select distinct(`dispatching_base_number`), 
                    sum(`trips`) as cnt from uber group by `dispatching_base_number` 
                    order by cnt desc""").show()

    sc.stop()

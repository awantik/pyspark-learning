"""Calculates the word count of the given file.

the file can be local or if you setup cluster.

It can be hdfs file path"""

## Imports

from pyspark import SparkConf, SparkContext

from operator import add
import sys
## Constants
APP_NAME = " HelloWorld of Big Data"
##OTHER FUNCTIONS/CLASSES

def main(sc,filename):
   textRDD = sc.textFile(filename)
   words = textRDD.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1))
   wordcount = words.reduceByKey(add).collect()
   for wc in wordcount:
      print (wc[0],wc[1])

if __name__ == "__main__":

   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME)
   conf = conf.setMaster("local[*]")
   sc   = SparkContext(conf=conf)
   filename = sys.argv[1]
   # Execute Main functionality
   main(sc, filename)
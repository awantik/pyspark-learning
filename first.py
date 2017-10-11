from pyspark import SparkConf, SparkContext
from operator import add

def main(sc):
    rdd = sc.parallelize([100,300,400])
    s = rdd.reduce(lambda a,b: a + b)
    print (s)

if __name__ == "__main__":
    conf = SparkConf().setAppName("my-app")
    conf = conf.setMaster("local")
    sc = SparkContext(conf = conf)
    main(sc)

'''
-class: The entry point for your application (e.g.apache.spark.examples.SparkPi)
–master: The master URL for the cluster (e.g. spark://23.195.26.187:7077)
–deploy-mode: Whether to deploy your driver on the worker nodes (cluster) or 
locally as an external client (client) (default:client)*
–conf: Arbitrary Spark configuration property in key=value format. 
For values that contain spaces wrap “key=value” in quotes (as shown).
'''
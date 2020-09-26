from pyspark import SparkContext, SparkConf
import sys
import json
from operator import add
import string
import csv
import random
import math
import time
from itertools import combinations

def append(a, b):
	a.append(b)
	return a

b=83
hashNum = 35
numBands = 35


def LSH(userList):
    numRows = int(math.ceil(int(hashNum) / int(numBands)))
    result = list()
    for i,x in enumerate(range(0,len(userList),numRows)):
        userTuple = hash(tuple(userList[x:x+numRows]))
        result.append((i,userTuple))
    return result


def jaccardSimCal(x,signatureMatrixSet):
    item1 = signatureMatrixSet[x[0]]
    item2 = signatureMatrixSet[x[1]]
    intersection = len(item1 & item2)
    union = len(item1) + len(item2) - intersection
    return float(intersection)/float(union)


def genSignatureMatrix(x,userLen):
    a = [83, 96, 44, 82, 25, 5, 101, 114, 87, 77, 4, 24, 48, 3, 100, 31, 35, 85, 76, 28, 37, 65, 94, 67, 10, 107, 22, 32, 57, 109, 81, 111, 59, 11
6, 19]
    print(a)
    #b = random.sample(range(0, 2*63 - 1), hashNum)
    h1 = x.map(lambda x : (x[0], (83*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    h2 = x.map(lambda x : (x[0], (96*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    res = h1.join(h2).mapValues(list)
    for i in range(2,hashNum):
        h = x.map(lambda x : (x[0], (a[i]*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
        res = res.join(h).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
    signatureMatrix = res.sortByKey()
    return signatureMatrix


if __name__ == "__main__":
 start = time.time()
 inputFile = sys.argv[1]
 conf = SparkConf().setAppName("Task1").setMaster("local[*]")
 sc = SparkContext(conf = conf)
 sc.setLogLevel("ERROR")

 dataset = sc.textFile(inputFile).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"],x['business_id'])).persist()
 userIndexRDD = dataset.map(lambda kv: kv[0]).distinct().sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
 userLen = len(userIndexRDD)
 businessIndexRDD = dataset.map(lambda kv: kv[1]).distinct().sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
 reversed_index_bus_dict = {v: k for k, v in businessIndexRDD.items()}
 bidx_uidxs_dict = dataset.map(lambda kv: (userIndexRDD[kv[0]],businessIndexRDD[kv[1]]))
 signatureMatrix = genSignatureMatrix(bidx_uidxs_dict,userLen)
 bidx_uidxs_dict_set = dataset.map(lambda kv: (userIndexRDD[kv[0]],businessIndexRDD[kv[1]])).groupByKey().mapValues(set).collectAsMap()
 result = signatureMatrix.flatMap(lambda x: [(tuple(result),x[0]) for result in LSH(x[1])]).groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>1)
 candidatePairs = result.flatMap(lambda x: [pair for pair in combinations(x, 2)]).distinct()
 #print(candidatePairs.count())
 jaccardSim = candidatePairs.map(lambda x: (list(x),jaccardSimCal(x,bidx_uidxs_dict_set)))
 truePairs = jaccardSim.filter(lambda x: x[1]>=0.05).collect()
 
 triplet = list()
 for pair in truePairs:
    triplet.append({"b1": reversed_index_bus_dict[pair[0][0]],
                                       "b2": reversed_index_bus_dict[pair[0][1]],
                                       "sim": pair[1]})

 with open("task1.res", 'w+') as output_file:
  for item in triplet:
     output_file.writelines(json.dumps(item) + "\n")
  output_file.close()

 print("Duration: %d s." % (time.time() - start))



 





 

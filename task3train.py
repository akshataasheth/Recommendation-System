from pyspark import SparkContext, SparkConf
import sys
import json
from operator import add
import string
import csv
import random
import math
import time
import re
import collections
from itertools import combinations

def append(a, b):
	a.append(b)
	return a

b=83
hashNum = 30
numBands = 30


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
    a = [83, 96, 44, 82, 25, 5, 101, 114, 87, 77, 4, 24, 48, 3, 100, 31, 35, 85, 76, 28, 37, 65, 94, 67, 10, 107, 22, 32, 57, 109, 81, 111, 59, 11,
    6, 19]
    #b = random.sample(range(0, 2*63 - 1), hashNum)
    h1 = x.map(lambda x : (x[0], (83*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    h2 = x.map(lambda x : (x[0], (96*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    res = h1.join(h2).mapValues(list)
    for i in range(2,hashNum):
        h = x.map(lambda x : (x[0], (a[i]*x[1]+b) % userLen)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
        res = res.join(h).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
    signatureMatrix = res.sortByKey()
    return signatureMatrix

"""
def computeSimilarity(dict1, dict2):
    user1 = set(dict1.keys())
    user2 = set(dict2.keys())
    co_rated_user = list(user1 & user2)
    list1 = list()
    list2 = list()
    for user in co_rated_user:
        list1.append(dict1[user])
        list2.append(dict2[user])

    avg1 = sum(list1) / len(list1)
    avg2 = sum(list2) / len(list2)
    myresults = []
    myresultsden = []
    myresultsden1 = []

    for x, y in zip(list1, list2):
        myresults.append((x-avg1) * (y-avg2))
        num = sum(myresults)
    
    for x, y in zip(list1, list2):
        myresultsden.append((x-avg1) ** 2)
        myresultsden1.append((x-avg1) ** 2)
        den = math.sqrt(sum(myresultsden)) * math.sqrt(sum(myresultsden1))

    if num == 0:
        return 0
    if den == 0:
        return 0
    return num/den

"""

def computeSimilarity(dict1, dict2):
    co_rated_user = list(set(dict1.keys()) & (set(dict2.keys())))
    val1_list, val2_list = list(), list()
    [(val1_list.append(dict1[user_id]),
      val2_list.append(dict2[user_id])) for user_id in co_rated_user]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2), zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
    denominator = math.sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  math.sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator

def combineDict(dict_list):
  finalMap = {}
  for d in dict_list:
      finalMap.update(d)
  return finalMap

if __name__ == "__main__":
 start = time.time()
 #x=sys.argv[1]
 inputFile = "train_review.json"
 outputFile="task3.model"
 cf_type = sys.argv[1]
 conf = SparkConf().setMaster("local[*]")
 sc = SparkContext(conf = conf)
 sc.setLogLevel("ERROR")
 dataset = sc.textFile(inputFile).map(lambda x: json.loads(x))\
       .map(lambda x: (x['user_id'],x['business_id'],x['stars'])).persist()
  
 
 if cf_type=="item_based":
     bid_uid_rdd = dataset.map(lambda x: (x[1],(x[0],x[2]))).groupByKey().mapValues(lambda x: list(x)).filter(lambda x: len(x[1])>=3) \
                           .mapValues(lambda vals: [{uid_score[0]: uid_score[1]} for uid_score in vals]) \
                              .mapValues(lambda val: combineDict(val)).persist()
 
     candidatePairs = bid_uid_rdd.map(lambda x:x[0])
 
     bidMap = bid_uid_rdd.map(lambda bid_uid_score: {bid_uid_score[0]: bid_uid_score[1]}) \
                 .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
 
     triplet = candidatePairs.cartesian(candidatePairs) \
                 .filter(lambda pair: pair[0] < pair[1]) \
                 .filter(lambda pair: ((bidMap[pair[0]] is not None) and (bidMap[pair[1]] is not None))) \
                     .filter(lambda pair: (len(set(bidMap[pair[0]].keys()) & set(bidMap[pair[1]].keys())) >=3)) \
                                         .map(lambda pair: (pair, computeSimilarity(bidMap[pair[0]],
                                                         bidMap[pair[1]]))).filter(lambda x: x[1]>0) \
                 .map(lambda kv: {"b1": kv[0][0],
                                  "b2": kv[0][1],
                                  "sim": kv[1]}).collect()
 else:
  userIndexRDD = dataset.map(lambda kv: kv[0]).distinct().sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
  businessIndexRDD = dataset.map(lambda kv: kv[1]).distinct().sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
  userLen = len(businessIndexRDD)
  reversed_index_bus_dict = {v: k for k, v in userIndexRDD.items()}
  bidMap = dataset.map(lambda kv: (userIndexRDD[kv[0]],(businessIndexRDD[kv[1]],kv[2]))).groupByKey().mapValues(lambda x: list(x)).mapValues(lambda vals: [{uid_score[0]: uid_score[1]} for uid_score in vals]) \
                              .mapValues(lambda val: combineDict(val)).map(lambda bid_uid_score: {bid_uid_score[0]: bid_uid_score[1]}) \
                       .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
 
  bidx_uidxs_dict = dataset.map(lambda kv: (userIndexRDD[kv[0]],businessIndexRDD[kv[1]]))
  bidx_uidxs_dict_set = bidx_uidxs_dict.groupByKey().mapValues(set).collectAsMap()
 
  signatureMatrix = genSignatureMatrix(bidx_uidxs_dict,userLen)
  #print(signatureMatrix.take(2))
 
  result = signatureMatrix.flatMap(lambda x: [(tuple(result),x[0]) for result in LSH(x[1])]).groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>1)
  candidatePairs = result.flatMap(lambda x: [pair for pair in combinations(x, 2)]).distinct() #pairs of uid
  #print(candidatePairs.count())
  
  jaccardSim = candidatePairs.map(lambda x: (list(x),jaccardSimCal(x,bidx_uidxs_dict_set)))
  truePairs = jaccardSim.filter(lambda x: x[1]>=0.01).filter(lambda pair: (len(set(bidMap[pair[0][0]].keys()) & set(bidMap[pair[0][1]].keys())) >=3)) \
       .map(lambda pair: (pair[0], computeSimilarity(bidMap[pair[0][0]], bidMap[pair[0][1]]))).filter(lambda x: x[1]>0).collect()
         
  triplet = list()
  for pair in truePairs:
     triplet.append({"u1": reversed_index_bus_dict[pair[0][0]],
                                        "u2": reversed_index_bus_dict[pair[0][1]],
                                        "sim": pair[1]})
 
 with open("task3user.model", 'w+') as output_file:
  for item in triplet:
     output_file.writelines(json.dumps(item) + "\n")
  output_file.close()
 #
#


#with open("task3.model", 'w+') as output_file:
#    for item in candidate_pair.collect():
#      output_file.writelines(json.dumps(item) + "\n")
#    output_file.close()

print("--- Duration: %s seconds ---" % (time.time() - start))

            
#print(candidate_pair.take(2))

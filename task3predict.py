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

def returnPair(bid,bid_pair_sim_dict):
    res = {}
    for item in bid_pair_sim_dict.keys():
      if bid in list(item):
          res[item] = bid_pair_sim_dict[item]
    return res

def makePrediction(x,model_lines,avg_scores,model_type):
    result = list()
    if model_type =='item_based':
        for bid in list(x[1]):
            if x[0] < bid[0]:
                    k = tuple((x[0], bid[0]))
            else:
                    k = tuple((bid[0], x[0]))
            item = tuple((bid[1], model_lines.get(k, 0)))
            result.append(item)

        score_sim_list = sorted(result, key=lambda item: item[1], reverse=True)[:3]
        numerator = sum(map(lambda item: item[0] * item[1], score_sim_list))
        if numerator == 0:
                return tuple((x[0], avg_scores.get(x[0], 3.823989)))
        denominator = sum(map(lambda item: abs(item[1]), score_sim_list))
        if denominator == 0:
                return tuple((x[0], avg_scores.get(x[0], 3.823989)))
        return tuple((x[0], numerator / denominator))
    else:
        for uid in list(x[1]):
            if x[0] < uid[0]:
                    k = tuple((x[0], uid[0]))
            else:
                    k = tuple((uid[0], x[0]))
            avgScore = avg_scores.get(uid[0],3.823989)
            item = tuple((uid[1],avgScore,model_lines.get(k,0)))
            result.append(item)
        num = sum(map(lambda item: (item[0] - item[1]) * item[2], result))
        if num ==0:
            return tuple((x[0], avg_scores.get(x[0], 3.823989)))
        den = sum(map(lambda item: abs(item[2]),result))
        if den == 0:
            return tuple((x[0], avg_scores.get(x[0], 3.823989)))
        avg_tuid_score = avg_scores.get(x[0], 3.823989)
        return tuple((x[0], avg_tuid_score+(num / den)))



if __name__ == "__main__":
    start = time.time()
    inputFile = sys.argv[1]
    modelFile = sys.argv[3]
    testFile = sys.argv[2]
    outputFile = sys.argv[4]
    avgBusinessFile = "../resource/asnlib/publicdata/business_avg.json"
    avgUserFile = "../resource/asnlib/publicdata/user_avg.json"
    cf_type = sys.argv[5]
    conf = SparkConf().setMaster("local[*]") \
           .setAppName("task2").set("spark.executor.memory", "4g") \
           .set("spark.driver.memory", "4g")
    sc = SparkContext(conf = conf)
    dataset = sc.textFile(inputFile).map(lambda row: json.loads(row)) \
           .map(lambda kv: (kv['user_id'], kv['business_id'], kv['stars'])).persist()
    

    if cf_type=="item_based":
        model_lines = sc.textFile(modelFile) \
                   .map(lambda row: json.loads(row)) \
                   .map(lambda kvv: {(kvv['b1'], kvv['b2']):kvv['sim']}) \
                   .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        test_lines = sc.textFile(testFile).map(lambda row: json.loads(row)) \
                   .map(lambda kv: (kv['user_id'], kv['business_id'])) \
                   .filter(lambda uid_bid: uid_bid[0] != -1 and uid_bid[1] != -1)  

        train_lines = dataset.map(lambda x: (x[0],(x[1],x[2]))).groupByKey().collectAsMap()

        avg_business_dict = sc.textFile(avgBusinessFile).map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()) \
                   .collectAsMap()


        #mixed
        test_train = test_lines.map(lambda x : (x[0],(x[1], train_lines[x[0]]))).mapValues(list)
        prediction = test_train.mapValues(lambda x: makePrediction(tuple(x),model_lines,avg_business_dict,'item_based')).map(lambda kvv: {"user_id": kvv[0],
                                     "business_id": kvv[1][0],
                                     "stars": kvv[1][1]})
    
    else:
        model_lines = sc.textFile(modelFile) \
                .map(lambda row: json.loads(row)) \
                .map(lambda kvv: {(kvv['u1'], kvv['u2']):kvv['sim']}) \
                .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
        
        train_lines = dataset.map(lambda x: (x[1],(x[0],x[2]))).groupByKey().collectAsMap()
        
        test_lines = sc.textFile(testFile).map(lambda row: json.loads(row)) \
                .map(lambda kv: (kv['business_id'], kv['user_id'])) \
                .filter(lambda uid_bid: uid_bid[0] != -1 and uid_bid[1] != -1)
    
        avg_user_dict = sc.textFile(avgUserFile).map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: x.items()) \
                .collectAsMap()  
    
        test_train = test_lines.map(lambda x : (x[0],(x[1], train_lines.get(x[0],-1)))).filter(lambda x: x[1][1] != -1 ).mapValues(list)
        prediction = test_train.mapValues(lambda x: makePrediction(tuple(x),model_lines,avg_user_dict,'user_based')).map(lambda kvv: {"user_id": kvv[1][0],
                                  "business_id": kvv[0],
                                  "stars": kvv[1][1]})
    
    with open(outputFile, 'w+') as output_file:
     for item in prediction.collect():
        output_file.writelines(json.dumps(item) + "\n")
     output_file.close()
    

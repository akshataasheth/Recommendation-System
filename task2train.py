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

start_time = time.time()

def formatResult(data, type, keys):
    result = list()
    if isinstance(data, dict):
        for key, val in data.items():
            result.append({"type": type,keys[0]: key,keys[1]: val})
    elif isinstance(data, list):
        for kv in data:
            for key, val in kv.items():
                result.append({"type": type,keys[0]: key,keys[1]: val})
    return result

def joinList(list1, list2):
    result = list(list1)
    result.extend(list2)
    return result

def splitTextAndRemove(texts, stop_words):
    word_list = list()
    for text in texts:
        text = text.translate(str.maketrans('', '', string.digits + string.punctuation))
        word_list.extend(
            list(filter(lambda word: word not in stop_words
                                     and word != ''
                                     and word not in string.ascii_lowercase,
                        re.split(r"[~\s\r\n]+", text))))

    return word_list


def countWords(bid,wordList):
  count = {}
  for word in wordList:
      if word not in count.keys():
          count[word] = 1
      else:
          count[word] += count[word] + 1
  itemMaxValue = max(count.values())
  return [tuple(((bid,k),float(v/itemMaxValue))) for k,v in count.items()]

  

if __name__ == "__main__":
 start = time.time()
 inputFile = sys.argv[1]
 stopwords = sys.argv[2]
 outputFile = sys.argv[3]
 conf = SparkConf().setMaster("local[*]") \
        .setAppName("task2").set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
 sc = SparkContext(conf = conf)
 #sc.setLogLevel("ERROR")
 stop_words = set()
 for line in open(stopwords):
   stop_words.add(line.strip())
 dataset = sc.textFile(inputFile).map(lambda x: json.loads(x))\
       .map(lambda x: (x['user_id'],x['business_id'],x['text']))


 #(bid,list of words)
 reviewRDD = dataset.map(lambda x: (x[1],str(x[2].encode('utf-8')).lower())).groupByKey() \
        .mapValues(lambda texts: splitTextAndRemove(list(texts), stop_words))
        
 dummyRDD = reviewRDD.count()

 #(bid,word),tfvalue))
 businessRDD = reviewRDD.flatMap(lambda x: (countWords(x[0],list(x[1])))).cache()
 
 IDFRDD = businessRDD.map(lambda words: (words[0][1],words[0][0])).groupByKey().mapValues(lambda bids: list(set(bids))) \
    .flatMap(lambda x: [((bid, x[0]),math.log(dummyRDD / len(x[1]), 2)) for bid in x[1]])
 
 tfidfRDD = businessRDD.leftOuterJoin(IDFRDD)\
        .mapValues(lambda tf_idf: tf_idf[0] * tf_idf[1]) \
        .map(lambda bid_word_val: (bid_word_val[0][0],
                                  (bid_word_val[0][1], bid_word_val[1]))) \
        .groupByKey() \
        .mapValues(lambda val: sorted(list(val), reverse=True,
                                      key=lambda item: item[1])[:200]) \
        .mapValues(lambda word_vals: [item[0] for item in word_vals])
        
 wordIndex = tfidfRDD \
        .flatMap(lambda kv: [(word, 1) for word in kv[1]]) \
        .groupByKey().map(lambda kv: kv[0]).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
        
 businessProfile = tfidfRDD.mapValues(lambda words: [wordIndex[word] for word in words]) \
        .map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()   

 model = list()
   

 model.extend(formatResult(businessProfile, 'business_profile',
                                 keys=['business_index', 'business_profile']))


 userProfile = dataset.map(lambda x: (x[0], x[1])) \
        .groupByKey().map(lambda x: (x[0], list(set(x[1])))).flatMapValues(lambda bids: [businessProfile[bid] for bid in bids]).reduceByKey(joinList).filter(lambda ids: len(ids[1]) > 1)\
        .map(lambda ids: {ids[0]: list(set(ids[1]))}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
        
       
        
 model.extend(formatResult(userProfile, 'user_profile',
                                 keys=['user_index', 'user_profile'])) 


 with open(outputFile, 'w+') as output_file:
   for item in model:
      output_file.writelines(json.dumps(item) + "\n")
   output_file.close()

 print("--- Duration: %s seconds ---" % (time.time() - start_time))

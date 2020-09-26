from pyspark import SparkConf, SparkContext
import json
import math
import sys
import time


def cosineSimilarity(item1, item2):
    if len(item1) != 0 and len(item2) != 0:
        userProfile = set(item1)
        businessProfile = set(item1)
        num = len(userProfile.intersection(businessProfile))
        den = math.sqrt(len(userProfile)) * math.sqrt(len(businessProfile))
        return num / den
    else:
        return 0.0


if __name__ == '__main__':
    start = time.time()
    testFile = sys.argv[1]
    modelFile = sys.argv[2]
    outputFile = sys.argv[3]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task3_predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    modelLines = sc.textFile(modelFile) \
        .map(lambda row: json.loads(row))

    user_profile = modelLines.filter(lambda kv: kv['type'] == 'user_profile') \
        .map(lambda kv: {kv['user_index']: kv['user_profile']}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()


    business_profile = modelLines.filter(lambda kv: kv['type'] == 'business_profile') \
        .map(lambda kv: {kv['business_index']: kv['business_profile']}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
        

    prediction = sc.textFile(testFile).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id'])) \
        .filter(lambda ids: ids[0] != -1 and ids[1] != -1) \
        .map(lambda x: ((x), cosineSimilarity(user_profile.get(x[0], set()),
                                                       business_profile.get(x[1], set())))) \
        .filter(lambda kv: kv[1] > 0.01) \
        .map(lambda kv: {"user_id": kv[0][0],
                         "business_id": kv[0][1],
                         "sim": kv[1]})
    #print(predict_result.take(2))
    result = prediction.collect()
    
    with open(outputFile, 'w+') as output_file:
        for item in result:
            output_file.writelines(json.dumps(item) + "\n")
        output_file.close()
    #export2File(predict_result.collect(), output_file_path)
    print("Duration: %d s." % (time.time() - start))
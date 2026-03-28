import math
def calc_accuracy(logs):
    return sum(log['actual'] == log['prediction'] for log in logs)/len(logs)

def latency_p95(logs):
    latencies = sorted([log['latency_ms'] for log in logs])
    return latencies[math.ceil(len(logs)*0.95) - 1]
    
def calc_aggrement_rate(production_log, shadow_log):
    actual = {
        log['input_id']: log['prediction']
        for log in production_log
    }

    predict = {
        log['input_id']: log['prediction']
        for log in shadow_log
    }
    
    return  sum([actual[input_id] == predict[input_id] for input_id in actual]) / len(actual)
    
def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    shadow_accuracy = calc_accuracy(shadow_log)
    production_accuracy = calc_accuracy(production_log)
    shadow_latency_p95 = latency_p95(shadow_log)
    accuracy_gain = shadow_accuracy - production_accuracy
    aggrement_rate = calc_aggrement_rate(production_log, shadow_log)

    promote = accuracy_gain>=criteria['min_accuracy_gain'] and shadow_latency_p95 <= criteria['max_latency_p95'] and aggrement_rate >= criteria['min_agreement_rate']
    
    return {'promote':promote,
             'metrics':{
                 "shadow_accuracy": shadow_accuracy,
                 "production_accuracy":production_accuracy,
                 "accuracy_gain": accuracy_gain,
                 "shadow_latency_p95": shadow_latency_p95,
                 'agreement_rate': aggrement_rate
             }}
   
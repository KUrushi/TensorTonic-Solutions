def retraining_policy(daily_stats, config):
    days_since_retrain = 0
    budget = config['budget']
    retraining_days = []
    # 初回実行かどうかを管理するフラグ、または前回の実行日を保持する
    last_retrain_day = None
    
    for daily_stat in daily_stats:
        # 1. 毎日 1 ずつ増加させる
        days_since_retrain += 1
        
        # 2. トリガー条件の判定 (厳密な比較)
        drift_trigger = daily_stat['drift_score'] > config['drift_threshold']
        perf_trigger = daily_stat['performance'] < config['performance_threshold']
        stale_trigger = days_since_retrain >= config['max_staleness']
        
        # 3. 制約条件の判定

        cooldown_ok = (last_retrain_day is None) or (days_since_retrain >= config['cooldown'])
        budget_ok = budget >= config['retrain_cost']
        
        # 4. 再学習の実行判断
        if (drift_trigger or perf_trigger or stale_trigger) and cooldown_ok and budget_ok:
            retraining_days.append(daily_stat['day'])
            budget -= config['retrain_cost']
            # 5. 状態のリセット
            days_since_retrain = 0
            last_retrain_day = daily_stat['day']
            
    return retraining_days
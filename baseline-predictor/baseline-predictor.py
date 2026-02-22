
def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    # Write code here
    user_mean = []
    item_mean = []
    global_totals = 0
    global_counts = 0

    item_counts = len(ratings_matrix[0])
    items_scores = {
        "total_ratings": [0.0 for _ in range(item_counts)],
        "effective_counts": [0.0 for _ in range(item_counts)]
    }
    
    for user_ratings in ratings_matrix:
        effective_rating_counts = 0
        total = 0
        for i in range(len(user_ratings)):
            if user_ratings[i] != 0:
                total += user_ratings[i]
                effective_rating_counts += 1
                items_scores['total_ratings'][i] += user_ratings[i]
                items_scores['effective_counts'][i] += 1

        global_totals += total
        global_counts += effective_rating_counts
        user_mean.append(total / effective_rating_counts)

    
    item_mean = [items_scores['total_ratings'][i]/items_scores['effective_counts'][i]
                for i in range(item_counts)]

                
    global_means = global_totals / global_counts

    outputs = []
    for u, i in target_pairs:
        outputs.append(global_means + (user_mean[u] - global_means) + (item_mean[i] - global_means))

    return outputs
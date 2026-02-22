def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    output = list(ratings_matrix)
    user_mean = []
    item_mean = []
    item_counts = len(ratings_matrix[0])
    items_scores = {
        "total_ratings": [0.0 for _ in range(item_counts)],
        "effective_counts": [0.0 for _ in range(item_counts)]
    }

    zero_index = []
    for user_index, user_ratings in enumerate(ratings_matrix):
        effective_rating_counts = 0
        total = 0
        for i in range(len(user_ratings)):
            if user_ratings[i] != 0:
                total += user_ratings[i]
                effective_rating_counts += 1
                items_scores['total_ratings'][i] += user_ratings[i]
                items_scores['effective_counts'][i] += 1
            else:
                zero_index.append((user_index, i))

        score = total / effective_rating_counts if effective_rating_counts != 0 else 0.0
        user_mean.append(score)

    for i in range(item_counts):
        score = items_scores['total_ratings'][i]/items_scores['effective_counts'][i] if items_scores['effective_counts'][i] != 0 else 0.0
        item_mean.append(score)

    if not zero_index:
        return output


    
    for u, i in zero_index:
        if mode == "user":
            output[u][i] = user_mean[u]
        else:
            output[u][i] = item_mean[i]

    return output
        
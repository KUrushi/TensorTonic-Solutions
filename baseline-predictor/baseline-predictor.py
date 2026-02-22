def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    # Write code here
    user_mean = []
    item_mean = []
    global_means = []
    for user_ratings in ratings_matrix:
        effective_rating_counts = 0
        total = 0
        for i in range(len(user_ratings)):
            if user_ratings[i] != 0:
                total += user_ratings[i]
                effective_rating_counts += 1
                global_means.append(user_ratings[i])
                
        user_mean.append(total / effective_rating_counts)

    for item_index in range(len(ratings_matrix[0])):
        effective_rating_counts = 0
        total = 0
        for i in range(len(ratings_matrix)):
            if ratings_matrix[i][item_index] != 0:
                total += ratings_matrix[i][item_index]
                effective_rating_counts += 1

        item_mean.append(total/effective_rating_counts)
                
    global_means = sum(global_means) / len(global_means)

    outputs = []
    for u, i in target_pairs:
        outputs.append(global_means + (user_mean[u] - global_means) + (item_mean[i] - global_means))

    return outputs
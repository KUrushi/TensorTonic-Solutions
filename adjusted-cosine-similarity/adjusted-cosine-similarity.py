def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    rating_user_mean = []
    for ratings in ratings_matrix:
        denominator = len(ratings) - ratings.count(0)
        rating_user_mean.append(sum(ratings) / denominator)
    
    if rating_user_mean.count(0) == len(rating_user_mean):
      return 0.0
      
    denominator_left = 0.0
    denominator_right = 0.0
    numerator = 0.0
    
    for user_ratings, rating_mean in zip(ratings_matrix, rating_user_mean):
      if user_ratings[item_i] == 0 or user_ratings[item_j] == 0:
        continue
          
      rating_ui = user_ratings[item_i] - rating_mean
      rating_uj = user_ratings[item_j] - rating_mean
      numerator += (rating_ui * rating_uj)
      denominator_left += (rating_ui ** 2)
      denominator_right += (rating_uj ** 2)

    if denominator_left == 0 or denominator_right == 0:
        return 0.0

    return numerator / (denominator_left ** 0.5 * denominator_right ** 0.5)
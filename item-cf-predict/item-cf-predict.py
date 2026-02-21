def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    # Write code here

    a = 0
    b = 0
  
    for i in range(len(user_ratings)):
        if i != target and user_ratings[i] != 0 and item_similarities[i] >0:
            a += item_similarities[i] * user_ratings[i]
            b += item_similarities[i]
    if b == 0:
      return 0.0
    return a / b
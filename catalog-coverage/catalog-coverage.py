def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    unique_items = set()
    for recommendation in recommendations:
        for r in recommendation:
            unique_items.add(r)

    return len(unique_items) / n_items
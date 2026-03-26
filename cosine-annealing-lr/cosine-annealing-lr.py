import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here
    return min_lr + (base_lr - min_lr) / 2 * (1 + np.cos(np.pi * current_step / total_steps))
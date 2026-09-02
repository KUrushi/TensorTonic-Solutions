# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# KUrushi's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/kurushi.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Adjusted Cosine Similarity | Compute adjusted cosine similarity for item-based collaborative filtering using co-rated users and mean-centered ratings. | https://www.tensortonic.com/problems/adjusted-cosine-similarity |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Angle Between 3D Vectors | Compute the angle between two 3D vectors in NumPy with clamped cosine values and safe handling of zero norms. | https://www.tensortonic.com/problems/angle-between-3d |
| Autocorrelation | Compute normalized time-series autocorrelation across a requested range of lags, including constant-series handling. | https://www.tensortonic.com/problems/autocorrelation |
| Baseline Predictor | Predict collaborative-filtering ratings from the global mean plus user and item rating biases. | https://www.tensortonic.com/problems/baseline-predictor |
| Batch Shuffling & Mini-Batch Generator | Create shuffled mini-batches from NumPy feature and target arrays with reproducible ordering and final-batch handling. | https://www.tensortonic.com/problems/batch-generator |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Binary Focal Loss | Compute binary focal loss from predicted probabilities with class balancing, focusing strength, and stable logarithms. | https://www.tensortonic.com/problems/binary-focal-loss |
| Binning | Assign numeric values to ordered bins using supplied boundaries while handling values at interval edges. | https://www.tensortonic.com/problems/binning |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| Catalog Coverage | Measure recommendation catalog coverage as the fraction of available items appearing across user recommendation lists. | https://www.tensortonic.com/problems/catalog-coverage |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Cohen's Kappa | Calculate Cohen's kappa from two label sequences by comparing observed agreement with chance agreement. | https://www.tensortonic.com/problems/cohens-kappa |
| Compute Confusion Matrix with Normalization | Build a multiclass confusion matrix and optionally normalize counts by true-class rows or predicted-class columns. | https://www.tensortonic.com/problems/confusion-matrix-norm |
| Implement Contrastive Loss (Siamese) | Implement Siamese-network contrastive loss using pair labels, embedding distances, and a separation margin. | https://www.tensortonic.com/problems/contrastive-loss |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Cosine Embedding Loss | Compute cosine embedding loss for similar and dissimilar vector pairs using labels and a configurable margin. | https://www.tensortonic.com/problems/cosine-embedding-loss |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Cyclic Encoding | Encode periodic numeric features as sine and cosine coordinates using a specified cycle length. | https://www.tensortonic.com/problems/cyclic-encoding |
| Implement Dice Loss | Compute Dice loss for segmentation predictions using overlap, total mass, and a numerical smoothing term. | https://www.tensortonic.com/problems/dice-loss |
| Differencing | Transform a time series into lagged differences while preserving the requested differencing interval. | https://www.tensortonic.com/problems/differencing |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| ε-Greedy Action Selection | Select a reinforcement-learning action with epsilon-greedy exploration using action values and controlled randomness. | https://www.tensortonic.com/problems/epsilon-greedy |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Feature Store Lookup | Combine stored offline and request-time features in input order, using defaults for unknown user IDs. | https://www.tensortonic.com/problems/feature-store-lookup |
| Implement Focal Loss | Compute mean binary focal loss from predicted probabilities using a configurable focusing parameter. | https://www.tensortonic.com/problems/focal-loss |
| Gaussian Blur Kernel | Generate a normalized 2D Gaussian blur kernel from an odd kernel size and positive standard deviation. | https://www.tensortonic.com/problems/gaussian-blur-kernel |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Geometric Probability Mass Function & Mean | Compute the geometric distribution probability mass and mean from a valid success probability. | https://www.tensortonic.com/problems/geometric-pmf-mean |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Impute Missing Values (mean/median) | Impute missing numeric values column-wise with either the mean or median while leaving observed values unchanged. | https://www.tensortonic.com/problems/impute-missing |
| Implement InfoNCE Loss | Compute InfoNCE contrastive loss from query and key embeddings using temperature-scaled similarities. | https://www.tensortonic.com/problems/info-nce-loss |
| Item-Based CF Prediction | Predict a user-item rating with item-based collaborative filtering from rated neighbors and item similarities. | https://www.tensortonic.com/problems/item-cf-predict |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Label Smoothing Loss | Compute multiclass cross-entropy with label smoothing by distributing target mass across all classes. | https://www.tensortonic.com/problems/label-smoothing-loss |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Learning Rate Scheduler (Linear Decay) | Compute a linearly decaying learning rate across training steps between configured start and end values. | https://www.tensortonic.com/problems/linear-lr-scheduler |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Factorization SGD Step | Perform one matrix-factorization SGD update for a rated user-item pair with latent factors and regularization. | https://www.tensortonic.com/problems/matrix-factorization-sgd-step |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Monte Carlo Policy Evaluation | Estimate state values from complete Monte Carlo episodes by averaging discounted returns for each visited state. | https://www.tensortonic.com/problems/mc-policy-evaluation |
| Mean Rating Imputation | Fill missing user-item ratings with the required row or column mean while preserving observed ratings. | https://www.tensortonic.com/problems/mean-rating-imputation |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| NDCG (Normalized Discounted Cumulative Gain) | Calculate normalized discounted cumulative gain at K from ranked relevance scores and their ideal ordering. | https://www.tensortonic.com/problems/ndcg |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Novelty Score | Measure recommendation novelty from item popularity by averaging self-information across recommended items. | https://www.tensortonic.com/problems/novelty-score |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| Popularity Ranking | Rank items by interaction popularity with deterministic ordering for equal-frequency items. | https://www.tensortonic.com/problems/popularity-ranking |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Prioritized Experience Replay | Compute prioritized replay sampling probabilities and normalized importance weights from transition priorities. | https://www.tensortonic.com/problems/priority-replay-sample |
| Random Forest Majority Vote | Combine multiple decision-tree predictions with majority voting and deterministic handling of tied classes. | https://www.tensortonic.com/problems/random-forest-vote |
| Rating Normalization | Normalize user-item ratings around the required mean while preserving missing-rating markers. | https://www.tensortonic.com/problems/rating-normalization |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Retraining Trigger Design | Evaluate production monitoring signals and decide whether they satisfy the configured model-retraining policy. | https://www.tensortonic.com/problems/retraining-trigger-design |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Robust Scaling | Scale numeric features using their median and interquartile range with constant-spread handling. | https://www.tensortonic.com/problems/robust-scaling |
| Rolling Standard Deviation | Compute rolling standard deviation over complete time-series windows using the required variance convention. | https://www.tensortonic.com/problems/rolling-standard-deviation |
| SARSA Update | Perform one on-policy SARSA action-value update from the observed reward and next selected action. | https://www.tensortonic.com/problems/sarsa-update |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Shadow Deployment Evaluation | Compare shadow and production model outcomes using the evaluation criteria defined by the problem. | https://www.tensortonic.com/problems/shadow-deployment-evaluation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Top-K Recommendations | Return each user's highest-scoring unseen items with deterministic ranking and a configurable result limit. | https://www.tensortonic.com/problems/top-k-recommendations |
| Detect Train-Serving Skew | Detect train-serving skew by comparing offline and online feature values under configured tolerances. | https://www.tensortonic.com/problems/train-serving-skew |
| Implement Triplet Loss | Compute triplet loss from anchor, positive, and negative embeddings using distances and a margin. | https://www.tensortonic.com/problems/triplet-loss |
| User-Based CF Prediction | Predict a user-item rating from similar users' ratings with neighborhood weighting and mean adjustment. | https://www.tensortonic.com/problems/user-based-cf-prediction |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Warmup + Linear Decay LR Schedule | Compute a learning-rate schedule with linear warmup followed by linear decay across training steps. | https://www.tensortonic.com/problems/warmup-decay-lr |
| Implement Wasserstein Critic Loss | Compute Wasserstein critic loss as the difference between mean fake and real critic scores. | https://www.tensortonic.com/problems/wasserstein-critic-loss |
| Weighted Moving Average | Compute a weighted moving average over complete time-series windows using normalized supplied weights. | https://www.tensortonic.com/problems/weighted-moving-average |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VAE Decoder | Build a variational autoencoder decoder that maps sampled latent vectors back to reconstructed input probabilities. | https://www.tensortonic.com/research/vae/vae-decoder |
| ELBO Loss Function | Implement the VAE evidence lower bound from reconstruction loss and KL divergence with configurable weighting. | https://www.tensortonic.com/research/vae/vae-elbo-loss |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| KL Divergence Loss | Compute the VAE KL-divergence term between a diagonal Gaussian posterior and the standard normal prior. | https://www.tensortonic.com/research/vae/vae-kl-divergence |
| Reparameterization Trick | Implement the VAE reparameterization trick by sampling latent vectors from the predicted mean and log variance. | https://www.tensortonic.com/research/vae/vae-reparameterization |
| Perceptron | Train a binary perceptron from zero-initialized weights using ordered samples, step predictions, and error-correction updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-perceptron |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/study-plans/cracking-inference/inference-grouped-query-attention |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-head-attention |
| Implement Multi-Head Latent Attention (MLA) | Implement Multi-Head Latent Attention (MLA), and return both the projected attention output and the compressed latent tensor. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-head-latent-attention |
| Implement Multi-Query Attention (MQA) | Implement multi-query attention with separate query heads, shared key-value heads, optional masking, and output projection. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-query-attention |
| Implement Rotary Position Embeddings for Decoding | Apply rotary position embeddings to query and key tensors at supplied absolute decoding positions. | https://www.tensortonic.com/study-plans/cracking-inference/inference-rotary-position-embeddings |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/study-plans/cracking-inference/inference-scaled-dot-product-attention |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-linear-regression-from-scratch |
| Logistic Regression from Scratch | Train binary logistic regression from scratch using sigmoid probabilities, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-logistic-regression |
| Bellman Expectation Equation | The Bellman expectation equation is the cornerstone of policy evaluation in reinforcement learning. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-expectation-equation |
| Bellman Optimality Equation | Apply one Bellman optimality backup by maximizing expected immediate reward plus discounted next-state value. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-optimality-equation |
| Discounted Returns | Compute reverse-time discounted returns from a reward sequence using a configurable discount factor and terminal bootstrap. | https://www.tensortonic.com/study-plans/cracking-rl/rl-discounted-returns |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with explicit handling for zero-norm inputs. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-cosine-similarity |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-dot-product |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors from the square root of summed squared differences. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-euclidean-distance |
| Gram-Schmidt Orthogonalization | Given k linearly independent vectors in R^n, the Gram-Schmidt process builds an orthonormal basis that spans exactly the same space. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-gram-schmidt |
| Hadamard Product | Compute elementwise multiplication between two same-shaped NumPy matrices to produce their Hadamard product. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-hadamard-product |
| Linear Combination | Compute a weighted linear combination of equal-length NumPy vectors using one aligned scalar coefficient per vector. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-linear-combination |
| Matrix Determinant | Compute the determinant of a square NumPy matrix as a scalar measure of invertibility and volume scaling. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-determinant |
| Matrix Multiply | Multiply compatible NumPy matrices and preserve the dtype produced by NumPy type-promotion rules. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-multiply |
| Matrix Rank | Compute the rank of a rectangular NumPy matrix from its number of linearly independent directions. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-rank |
| Matrix Trace | Compute the trace of a square NumPy matrix by summing its main-diagonal elements with numeric dtype support. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-trace |
| Matrix Transpose | Transpose a rectangular NumPy matrix by swapping its row and column axes without changing element values. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-transpose |
| Matrix-Vector Multiply | Multiply a NumPy matrix by a compatible vector, producing one row-wise dot product per output element. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-vector-multiply |
| Outer Product | Compute the NumPy outer product of two vectors as a matrix containing every pairwise element multiplication. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-outer-product |
| Solve Linear System | Solve an invertible square linear system for the unique vector satisfying the matrix equation. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-solve-linear-system |
| Vector Norms | Compute L1, L2, and infinity norms for a one-dimensional NumPy vector and return them in a float64 array. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-vector-norms |
| Vector Projection | The vector projection of u onto v is the component of u that lies exactly along the direction of v. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-vector-projection |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/kurushi)
<!-- tensortonic:end -->

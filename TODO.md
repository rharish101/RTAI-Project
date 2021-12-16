# TODOs

## Immediate
* Market research - find out about other groups' progress - Everyone

## Future
* PGD for generating test cases

## Possible Improvement Areas
* Weight connectivity differs from traditional NNs - all previous layers connected with current output
* Weights are passed through sigmoid before being used
  - PGD didn't improve
* Gradients are sparse and sparsity may shift
  - Similar to dropout
* No stochasticity in inputs
  - Add noise to gradients
* We want to overfit, not generalize
* Vanishing gradients:
  - Play around with loss functions

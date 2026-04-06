# Cognitive AI Sepal XGBoost

This project implements the following scheme:

1. `Per(alpha)` = observed sepal variables
2. `eta' = f(Per(alpha))` = internal representation learned by XGBoost
3. `P(y=1|x) = g(eta')` = class probability
4. `Act = 1(P >= tau)` = raw action
5. `Act* = h(Act, Pi)` = action corrected by reflexes
6. `Delta = E[l(Act, R)]` = action error
7. if `Delta > epsilon`, activation of `Pi_dyn`

## Default choices

* Dataset: Iris (`sklearn.datasets.load_iris`)
* Default features: `sepal length (cm)` and `sepal width (cm)`
* Default binary task: `setosa` vs rest
* Cognition: `XGBClassifier`
* Decision operator: threshold `tau` optimized on the training set
* Dynamic reflexes:

  * blocking within an uncertainty zone around `tau`
  * blocking atypical samples using a simple z-score

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Run without noise

```bash
python3 cognitive_ai_sepal_xgboost.py --pretty
```

## Run with noise handling

```bash
python cognitive_ai_sepal_xgboost.py --pretty --noise-fraction 0.35 --noise-std 1.25 --epsilon 0.04
```

Change the positive class:

```bash
python3 cognitive_ai_sepal_xgboost.py --positive-class virginica --pretty
```

Add petal features if needed:

```bash
python3 cognitive_ai_sepal_xgboost.py \
  --features "sepal length (cm)" "sepal width (cm)" "petal length (cm)" "petal width (cm)" \
  --pretty
```

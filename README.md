# Cognitive AI Sepal XGBoost

Ce projet implemente le schéma:

1. `Per(alpha)` = variables observees du sepale
2. `eta' = f(Per(alpha))` = representation interne apprises par XGBoost
3. `P(y=1|x) = g(eta')` = probabilite de classe
4. `Act = 1(P >= tau)` = action brute
5. `Act* = h(Act, Pi)` = action corrigee par reflexes
6. `Delta = E[l(Act, R)]` = erreur d'action
7. si `Delta > epsilon`, activation de `Pi_dyn`

## Choix par defaut

- Dataset: Iris (`sklearn.datasets.load_iris`)
- Features par defaut: `sepal length (cm)` et `sepal width (cm)`
- Tache binaire par defaut: `setosa` vs reste
- Cognition: `XGBClassifier`
- Operateur de decision: seuil `tau` optimise sur le train
- Reflexes dynamiques:
  - blocage dans une zone d'incertitude autour de `tau`
  - blocage des échantillons atypiques via un z-score simple

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Execution sans bruit

```bash
python3 cognitive_ai_sepal_xgboost.py --pretty
```

## Execution avec gestion du bruit


```bash
python cognitive_ai_sepal_xgboost.py --pretty --noise-fraction 0.35 --noise-std 1.25 --epsilon 0.04
```

Changer la classe positive:

```bash
python3 cognitive_ai_sepal_xgboost.py --positive-class virginica --pretty
```

Ajouter les petales si besoin:

```bash
python3 cognitive_ai_sepal_xgboost.py \
  --features "sepal length (cm)" "sepal width (cm)" "petal length (cm)" "petal width (cm)" \
  --pretty
```

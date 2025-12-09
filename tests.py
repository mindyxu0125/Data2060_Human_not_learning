import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.cart2 import DecisionTreeCART   

# ============================
#   Load dataset
# ============================
data = load_iris()
X, y = data.data, data.target

# Use fixed split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

for criterion in ["gini", "entropy"]:
    print(f"\nDataset: IRIS ({criterion.upper()})")
    print("=====================================")

    # ============================
    #   Train sklearn CART
    # ============================
    sk = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0
    )
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)

    # ============================
    #   Train OUR CART
    # ============================
    my = DecisionTreeCART(
        impurity=criterion,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0
    )
    my.fit(X_train, y_train)
    y_pred_my = my.predict(X_test)

    # ============================
    #   Compare
    # ============================
    exact = np.array_equal(y_pred_my, y_pred_sk)
    agreement = np.mean(y_pred_my == y_pred_sk)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    acc_my = accuracy_score(y_test, y_pred_my)

    print("Exact match:", exact)
    print(f"Prediction agreement: {agreement * 100:.2f}%")
    print(f"sklearn accuracy:     {acc_sk:.4f}")
    print(f"our CART accuracy:    {acc_my:.4f}")

    if exact:
        print("SUCCESS: Exact match achieved on the Iris dataset.")
    else:
        print("Mismatch detected — check tie-breaking / impurity implementation.")
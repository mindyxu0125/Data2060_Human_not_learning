"""
Compare cart2.DecisionTreeCART with sklearn's DecisionTreeClassifier
on multiple datasets and seeds.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.cart2 import DecisionTreeCART

# Datasets configuration
datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer(),
    'Digits': load_digits(),
}

seeds = [0, 1, 2, 3, 10, 20,30,33,50, 42, 99, 100, 1030, 1470, 2020,2060]
criteria = ['gini', 'entropy']

print('=' * 80)
print('Multi-dataset compatibility test: cart2.py vs sklearn')
print('=' * 80)
print()

# Store all results
all_results = {}

for dataset_name, dataset in datasets.items():
    print(f'\n{"=" * 80}')
    print(f'Dataset: {dataset_name}')
    print(f'{"=" * 80}')
    print(f'  Num samples: {dataset.data.shape[0]}')
    print(f'  Num features: {dataset.data.shape[1]}')
    print(f'  Num classes: {len(np.unique(dataset.target))}')
    print()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target,
        test_size=0.3,
        random_state=0,
        stratify=dataset.target
    )

    dataset_results = {}

    for criterion in criteria:
        exact_matches = 0
        total_tests = 0
        agreements = []
        acc_diffs = []

        for seed in seeds:
            # sklearn tree
            sk = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=seed
            )
            sk.fit(X_train, y_train)
            y_pred_sk = sk.predict(X_test)
            sk_acc = accuracy_score(y_test, y_pred_sk)

            # our tree
            our = DecisionTreeCART(
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                impurity=criterion,
                random_state=seed
            )
            our.fit(X_train, y_train)
            y_pred_our = our.predict(X_test)
            our_acc = accuracy_score(y_test, y_pred_our)

            # comparison
            exact_match = np.array_equal(y_pred_sk, y_pred_our)
            agreement = np.mean(y_pred_sk == y_pred_our)
            acc_diff = abs(sk_acc - our_acc)

            total_tests += 1
            if exact_match:
                exact_matches += 1

            agreements.append(agreement)
            acc_diffs.append(acc_diff)

        dataset_results[criterion] = {
            'exact_matches': exact_matches,
            'total': total_tests,
            'avg_agreement': np.mean(agreements),
            'min_agreement': np.min(agreements),
            'max_agreement': np.max(agreements),
            'avg_acc_diff': np.mean(acc_diffs),
            'max_acc_diff': np.max(acc_diffs),
        }

    all_results[dataset_name] = dataset_results

    # Print per-dataset results
    for criterion in criteria:
        res = dataset_results[criterion]
        print(f'  {criterion.upper()}:')
        print(f'    Exact matches: {res["exact_matches"]}/{res["total"]} '
              f'({res["exact_matches"]/res["total"]*100:.1f}%)')
        print(f'    Agreement: avg={res["avg_agreement"]*100:.2f}%, '
              f'min={res["min_agreement"]*100:.2f}%, '
              f'max={res["max_agreement"]*100:.2f}%')
        print(f'    Acc diff:  avg={res["avg_acc_diff"]:.4f}, '
              f'max={res["max_acc_diff"]:.4f}')
        print()

# Summary
print('\n' + '=' * 80)
print('Summary')
print('=' * 80)
print()

print(f'{"Dataset":<20} {"Criterion":<10} {"Exact Match":<15} '
      f'{"Avg Agreement":<18} {"Avg Acc Diff":<12}')
print('-' * 80)

for dataset_name in datasets.keys():
    for criterion in criteria:
        res = all_results[dataset_name][criterion]
        exact_pct = f'{res["exact_matches"]}/{res["total"]} ' \
                    f'({res["exact_matches"]/res["total"]*100:.0f}%)'
        agreement = f'{res["avg_agreement"]*100:.2f}%'
        acc_diff = f'{res["avg_acc_diff"]:.4f}'

        print(f'{dataset_name:<20} {criterion:<10} {exact_pct:<15} '
              f'{agreement:<18} {acc_diff:<12}')

print()
print('=' * 80)
print('Key findings')
print('=' * 80)

# Find best and worst cases by agreement
best_agreement = max(
    (dataset_name, criterion, res['avg_agreement'])
    for dataset_name, results in all_results.items()
    for criterion, res in results.items()
)

worst_agreement = min(
    (dataset_name, criterion, res['avg_agreement'])
    for dataset_name, results in all_results.items()
    for criterion, res in results.items()
)

most_exact = max(
    (dataset_name, criterion, res['exact_matches'], res['total'])
    for dataset_name, results in all_results.items()
    for criterion, res in results.items()
)

print(f'\nHighest average agreement: '
      f'{best_agreement[0]}, {best_agreement[1]} '
      f'→ {best_agreement[2]*100:.2f}%')
print(f'Lowest average agreement:  '
      f'{worst_agreement[0]}, {worst_agreement[1]} '
      f'→ {worst_agreement[2]*100:.2f}%')
print(f'Most exact matches:        '
      f'{most_exact[0]}, {most_exact[1]} '
      f'→ {most_exact[2]}/{most_exact[3]} '
      f'({most_exact[2]/most_exact[3]*100:.0f}%)')

# Global stats
all_agreements = [
    res['avg_agreement']
    for results in all_results.values()
    for res in results.values()
]

all_exact_rates = [
    res['exact_matches'] / res['total']
    for results in all_results.values()
    for res in results.values()
]

print(f'\nOverall average agreement: {np.mean(all_agreements)*100:.2f}%')
print(f'Overall average exact-match rate: {np.mean(all_exact_rates)*100:.1f}%')

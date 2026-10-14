"""Dataset-agnostic preprocessing, task inference and model evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score)


def suggest_task(y):
    """Heuristic only; users can always override."""
    values = y.dropna()
    if values.empty:
        return 'Classification'
    if not pd.api.types.is_numeric_dtype(values):
        return 'Classification'
    unique = values.nunique()
    if unique <= 10 and unique / len(values) <= 0.1:
        return 'Classification'
    return 'Regression'


def validate_columns(df, target, features):
    if not features:
        raise ValueError('Select at least one predictor column.')
    if target in features:
        raise ValueError('The target cannot also be a predictor.')
    if not df.columns.is_unique:
        raise ValueError('Duplicate column names are not supported.')
    if df[target].dropna().empty:
        raise ValueError('Target contains no observed values.')
    if len(df.dropna(subset=[target])) < 8:
        raise ValueError('At least 8 rows with non-missing target values are required.')


def preprocessor_for(X, scale=False):
    X = X.copy()
    numeric = X.select_dtypes(include='number').columns.tolist()
    other = [col for col in X.columns if col not in numeric]
    parts = []
    if numeric:
        steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
        if scale:
            steps.append(('scale', StandardScaler()))
        parts.append(('numeric', Pipeline(steps), numeric))
    if other:
        parts.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='(missing)', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), other))
    return ColumnTransformer(parts, remainder='drop')


def coerce_features(df, features):
    X = df[features].copy()
    for col in X:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype('object')
            X[col] = X[col].where(X[col].notna(), '(missing)').astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return X


def fit_supervised(df, target, features, task, algorithm, test_fraction=0.25,
                   max_depth=5, learning_rate=0.1, class_balance=False):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier, XGBRegressor

    validate_columns(df, target, features)
    valid = df.dropna(subset=[target]).copy()
    X = coerce_features(valid, features)
    if X.isna().all(axis=None):
        raise ValueError('No usable predictor values were found.')
    y_raw = valid[target]
    encoder = None
    if task == 'Classification':
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y_raw.astype(str)), index=valid.index)
        n_classes = len(encoder.classes_)
        if n_classes < 2:
            raise ValueError('Classification requires at least two target classes.')
        if n_classes > min(50, len(y)//3):
            raise ValueError('Too many target classes for reliable classification. Select a more suitable target or use regression.')
        counts = y.value_counts()
        n_test = int(np.ceil(len(y) * test_fraction))
        stratify = y if counts.min() >= 2 and n_test >= n_classes and (len(y)-n_test) >= n_classes else None
    else:
        y = pd.to_numeric(y_raw, errors='coerce')
        keep = y.notna() & np.isfinite(y)
        X, y = X.loc[keep], y.loc[keep]
        if len(y) < 8:
            raise ValueError('Regression needs at least 8 rows with numeric target values.')
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=stratify)
    if task == 'Classification':
        if y_train.nunique() < len(encoder.classes_):
            raise ValueError('Some classes are missing from training data. Use a larger dataset or different test fraction.')
        if algorithm == 'Logistic Regression':
            estimator = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced' if class_balance else None)
        elif algorithm == 'Random Forest':
            estimator = RandomForestClassifier(n_estimators=200, max_depth=max_depth, random_state=42, n_jobs=-1, class_weight='balanced' if class_balance else None)
        elif algorithm == 'XGBoost':
            cfg = {'objective': 'binary:logistic'} if len(encoder.classes_) == 2 else {'objective': 'multi:softprob', 'num_class':len(encoder.classes_)}
            estimator = XGBClassifier(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                                      random_state=42, n_jobs=2, tree_method='hist', **cfg)
        else:
            raise ValueError('Unknown classification algorithm.')
    else:
        if algorithm == 'Ridge Regression':
            estimator = Ridge(alpha=1.0)
        elif algorithm == 'Random Forest':
            estimator = RandomForestRegressor(n_estimators=200, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif algorithm == 'XGBoost':
            estimator = XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                                     random_state=42, n_jobs=2, tree_method='hist')
        else:
            raise ValueError('Unknown regression algorithm.')
    model = Pipeline([('preprocess', preprocessor_for(X_train, scale=algorithm in ('Logistic Regression','Ridge Regression'))),
                      ('estimator', estimator)])
    # XGBoost does not accept class_weight; use balanced sample weights instead.
    fit_kwargs = {}
    if task == 'Classification' and algorithm == 'XGBoost' and class_balance:
        from sklearn.utils.class_weight import compute_sample_weight
        fit_kwargs['estimator__sample_weight'] = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, **fit_kwargs)
    predicted = model.predict(X_test)
    output = pd.DataFrame({'Row ID': ['ROW-' + str(i).zfill(6) for i in range(1, len(X_test)+1)]}, index=X_test.index)
    if task == 'Classification':
        output['Actual'] = encoder.inverse_transform(y_test.to_numpy().astype(int))
        output['Predicted'] = encoder.inverse_transform(predicted.astype(int))
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
            output['Prediction confidence'] = probs.max(axis=1)
            if probs.shape[1] == 2:
                output['Positive class probability'] = probs[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average='weighted', zero_division=0)
        metrics = {'Accuracy': accuracy_score(y_test, predicted), 'Balanced accuracy': balanced_accuracy_score(y_test, predicted),
                   'Precision': precision, 'Recall': recall, 'F1': f1}
        matrix = confusion_matrix(y_test, predicted, labels=np.arange(len(encoder.classes_)))
        labels = [str(c) for c in encoder.classes_]
        report = pd.DataFrame(classification_report(y_test, predicted, labels=np.arange(len(labels)),
                        target_names=labels, output_dict=True, zero_division=0)).T.loc[labels]
        # The positive class is the second class in LabelEncoder's ordering.
        probabilities = probs[:, 1] if hasattr(model, 'predict_proba') and len(labels) == 2 else None
        binary_y = y_test.to_numpy() if len(labels) == 2 else None
    else:
        output['Actual'] = y_test.to_numpy()
        output['Predicted'] = predicted
        output['Residual'] = output['Actual'] - output['Predicted']
        metrics = {'MAE': mean_absolute_error(y_test, predicted),
                   'RMSE': np.sqrt(mean_squared_error(y_test, predicted)),
                   'R²': r2_score(y_test, predicted) if len(y_test) >= 2 else float('nan')}
        matrix, labels = None, None
        report, probabilities, binary_y = None, None, None
    return {'predictions': output.reset_index(drop=True), 'metrics': metrics,
            'matrix': matrix, 'labels': labels, 'task': task, 'algorithm': algorithm,
            'train_size': len(X_train), 'test_size': len(X_test),
            'class_report': report, 'binary_probabilities': probabilities, 'binary_actual': binary_y,
            'class_balance': class_balance}

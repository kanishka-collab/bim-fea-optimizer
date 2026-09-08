import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

def train_and_save_surrogate_model(csv_path="fea_dataset.csv", model_path="surrogate_model.pkl"):
    """
    Trains RandomForest surrogate models (Classifier for optimal beam/column section,
    Regressor for total steel weight and member forces) using structural dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found. Please generate it first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset '{csv_path}' with {len(df)} samples.")

    # Features: Height, Span X, Span Z, Point Load P, Distributed Load w
    feature_cols = ['H_m', 'Lx_m', 'Lz_m', 'point_load_P_kN', 'dist_load_w_kNm']
    X = df[feature_cols]

    # Target outputs
    y_beam_sec = df['optimal_beam_sec']
    y_col_sec = df['optimal_col_sec']
    y_weight = df['total_steel_weight_kg']
    y_beam_M = df['beam_M_u_kNm']
    y_col_P = df['col_P_u_kN']
    y_col_M = df['col_M_u_kNm']

    # Train/Test Split (80% train, 20% test)
    X_train, X_test, y_weight_train, y_weight_test = train_test_split(X, y_weight, test_size=0.2, random_state=42)
    _, _, y_beam_sec_train, y_beam_sec_test = train_test_split(X, y_beam_sec, test_size=0.2, random_state=42)
    _, _, y_col_sec_train, y_col_sec_test = train_test_split(X, y_col_sec, test_size=0.2, random_state=42)
    _, _, y_beam_M_train, y_beam_M_test = train_test_split(X, y_beam_M, test_size=0.2, random_state=42)
    _, _, y_col_P_train, y_col_P_test = train_test_split(X, y_col_P, test_size=0.2, random_state=42)

    # 1. Train Classifiers for Optimal Sections
    clf_beam = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_beam.fit(X_train, y_beam_sec_train)
    acc_beam = accuracy_score(y_beam_sec_test, clf_beam.predict(X_test))

    clf_col = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_col.fit(X_train, y_col_sec_train)
    acc_col = accuracy_score(y_col_sec_test, clf_col.predict(X_test))

    # 2. Train Regressors for Steel Weight and Forces
    reg_weight = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_weight.fit(X_train, y_weight_train)
    r2_weight = r2_score(y_weight_test, reg_weight.predict(X_test))

    reg_beam_M = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_beam_M.fit(X_train, y_beam_M_train)
    r2_beam_M = r2_score(y_beam_M_test, reg_beam_M.predict(X_test))

    reg_col_P = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_col_P.fit(X_train, y_col_P_train)
    r2_col_P = r2_score(y_col_P_test, reg_col_P.predict(X_test))

    print("\n==================================================")
    print("      AI SURROGATE MODEL TRAINING RESULTS         ")
    print("==================================================")
    print(f"  RandomForest Total Weight R^2 Score   : {r2_weight:.4f}")
    print(f"  RandomForest Beam M_u R^2 Score        : {r2_beam_M:.4f}")
    print(f"  RandomForest Column P_u R^2 Score      : {r2_col_P:.4f}")
    print(f"  Beam Section Classifier Accuracy       : {acc_beam * 100:.2f}%")
    print(f"  Column Section Classifier Accuracy     : {acc_col * 100:.2f}%")
    print("==================================================\n")

    # Package trained surrogate models into dictionary
    surrogate = {
        'feature_cols': feature_cols,
        'clf_beam': clf_beam,
        'clf_col': clf_col,
        'reg_weight': reg_weight,
        'reg_beam_M': reg_beam_M,
        'reg_col_P': reg_col_P,
        'metrics': {
            'r2_weight': round(r2_weight, 4),
            'r2_beam_M': round(r2_beam_M, 4),
            'r2_col_P': round(r2_col_P, 4),
            'acc_beam': round(acc_beam * 100, 2),
            'acc_col': round(acc_col * 100, 2)
        }
    }

    # Save to pkl file
    joblib.dump(surrogate, model_path)
    print(f"Surrogate model bundle successfully saved to '{model_path}'.")
    return surrogate

if __name__ == '__main__':
    train_and_save_surrogate_model()

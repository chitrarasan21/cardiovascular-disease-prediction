"""Student-style cardiovascular disease prediction analysis."""
from pathlib import Path
import argparse
import json
import warnings
import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/chitrarasanmacbookair/Downloads/cardio_train (1).csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, sep=";")
    raw_rows = len(df)
    df = df.drop(columns=["id"], errors="ignore").drop_duplicates().copy()
    # Age is stored in days; convert to years for readable analysis.
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df.drop(columns=["age"])
    # Remove physiologically impossible blood pressure and height/weight records.
    valid = (df.ap_hi.between(70, 250) & df.ap_lo.between(40, 200) &
             (df.ap_hi >= df.ap_lo) & df.height.between(120, 220) &
             df.weight.between(30, 200))
    before_filter = len(df)
    df = df.loc[valid].reset_index(drop=True)
    # BMI is a useful derived feature; retain the original measures too.
    df["bmi"] = df.weight / (df.height / 100) ** 2
    target = "cardio"
    features = [c for c in df.columns if c != target]
    X, y = df[features], df[target].astype(int)
    numeric = features
    preprocess = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), numeric)], remainder="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y)
    models = {
        "SVM": SVC(kernel="rbf", C=1, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    rows=[]; fitted={}
    for name, clf in models.items():
        pipe=Pipeline([("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        try: auc=roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1] if hasattr(pipe,"predict_proba") else pipe.decision_function(X_test))
        except Exception: auc=np.nan
        cv=cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy", n_jobs=1)
        rows.append({"Model":name,"Test accuracy":accuracy_score(y_test,pred),"ROC-AUC":auc,"3-fold CV accuracy":cv.mean()})
        fitted[name]=(pipe,pred)
    scores=pd.DataFrame(rows).sort_values("Test accuracy",ascending=False)
    scores.to_csv(out/"model_comparison.csv",index=False)
    best_name=scores.iloc[0]["Model"]
    best,pred=fitted[best_name]
    joblib.dump(best,out/"cardiovascular_model.joblib")
    (out/"model_metadata.json").write_text(json.dumps({"model":best_name,"features":features,"target":target,"test_accuracy":float(scores.iloc[0]["Test accuracy"]),"roc_auc":float(scores.iloc[0]["ROC-AUC"])},indent=2))
    # Summary tables
    (out/"classification_report.txt").write_text(classification_report(y_test,pred,target_names=["No disease","Disease"]))
    df.describe().T.to_csv(out/"descriptive_statistics.csv")
    corr=df.corr(numeric_only=True)
    corr.to_csv(out/"correlation_matrix.csv")
    # Visualizations: distributions, target, categories, relationships and models.
    fig,ax=plt.subplots(figsize=(6,4)); sns.countplot(data=df,x=target,ax=ax); ax.set(title="Cardiovascular disease class balance",xlabel="Cardio (0 = no disease, 1 = disease)",ylabel="Number of people"); fig.tight_layout(); fig.savefig(out/"target_balance.png",dpi=160); plt.close(fig)
    for col in ["age_years","height","weight","bmi","ap_hi","ap_lo"]:
        fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(data=df,x=col,hue=target,stat="density",common_norm=False,kde=True,element="step",ax=ax); ax.set_title(f"{col.replace('_',' ').title()} distribution by disease status"); fig.tight_layout(); fig.savefig(out/f"distribution_{col}.png",dpi=150); plt.close(fig)
    for col in ["gender","cholesterol","gluc","smoke","alco","active"]:
        rate=df.groupby(col,observed=True)[target].mean().reset_index()
        fig,ax=plt.subplots(figsize=(6,4)); sns.barplot(data=rate,x=col,y=target,ax=ax); ax.set(title=f"Disease proportion by {col}",ylabel="Proportion with disease",xlabel=col.title(),ylim=(0,1)); fig.tight_layout(); fig.savefig(out/f"disease_rate_{col}.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,9)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,ax=ax); ax.set_title("Feature correlation matrix"); fig.tight_layout(); fig.savefig(out/"correlation_matrix.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); sns.barplot(data=scores,x="Model",y="Test accuracy",ax=ax); ax.set_ylim(0,1); ax.tick_params(axis="x",rotation=20); ax.set_title("Test accuracy of five classifiers"); fig.tight_layout(); fig.savefig(out/"model_accuracy.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=["No disease","Disease"],cmap="Blues",ax=ax,colorbar=False); ax.set_title(f"Confusion matrix: {best_name}"); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    # Markdown data-driven findings for report.
    positives=int(y.sum()); n=len(y)
    lines=[f"Rows in original file: {raw_rows}",f"Rows after duplicate removal and plausibility filtering: {n} (removed {raw_rows-n})",f"Positive disease cases: {positives} ({positives/n:.1%})",f"Selected model: {best_name}",f"Selected model test accuracy: {scores.iloc[0]['Test accuracy']:.4f}",f"Selected model ROC-AUC: {scores.iloc[0]['ROC-AUC']:.4f}","", "Model comparison:",scores.to_string(index=False)]
    (out/"analysis_results.txt").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__=="__main__": main()

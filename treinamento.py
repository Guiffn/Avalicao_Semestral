from pathlib import Path
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "breast-cancer.data"
ARQUIVO_MODELO = PASTA_PROJETO / "breast_cancer_model.pkl"

COLUNAS = [
    "Class",
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]
ATRIBUTOS = COLUNAS[1:]
CLASSE = "Class"


def carregar_dados():
    dados = pd.read_csv(
        ARQUIVO_DADOS,
        header=None,
        names=COLUNAS,
        na_values=["?"],
        dtype=str,
    )
    return dados.dropna(subset=[CLASSE])


def criar_pipeline():
    codificador = ColumnTransformer(
        [
            (
                "categoricos",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=int,
                    ),
                ),
                ATRIBUTOS,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("codificador_categorico", codificador),
            ("balanceador", SMOTEN(random_state=42)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("normalizador", MinMaxScaler()),
            ("classificador", DecisionTreeClassifier(random_state=42)),
        ]
    )


def treinar():
    dados = carregar_dados()
    X = dados[ATRIBUTOS]
    y = dados[CLASSE]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    espacos_parametros = [
        {
            "classificador": [DecisionTreeClassifier(random_state=42)],
            "classificador__criterion": ["gini", "entropy", "log_loss"],
            "classificador__max_depth": [None, 3, 5, 8, 12],
            "classificador__min_samples_split": [2, 4, 8, 12],
            "classificador__min_samples_leaf": [1, 2, 4, 6],
        },
        {
            "classificador": [RandomForestClassifier(random_state=42)],
            "classificador__n_estimators": [100, 200],
            "classificador__criterion": ["gini", "entropy", "log_loss"],
            "classificador__max_depth": [None, 5, 10],
            "classificador__min_samples_leaf": [1, 2, 4],
            "classificador__max_features": ["sqrt", None],
        },
        {
            "classificador": [SVC()],
            "classificador__C": [0.1, 1, 10, 100],
            "classificador__kernel": ["linear", "rbf"],
            "classificador__gamma": ["scale", "auto"],
        },
    ]
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = RandomizedSearchCV(
        estimator=criar_pipeline(),
        param_distributions=espacos_parametros,
        n_iter=30,
        scoring="accuracy",
        cv=validacao,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    busca.fit(X_treino, y_treino)

    modelo_avaliado = busca.best_estimator_
    predicoes = modelo_avaliado.predict(X_teste)
    print("Quantidade de registros:", len(dados))
    print("Atributos ausentes:", int(dados[ATRIBUTOS].isna().sum().sum()))
    print("Melhor classificador:", type(modelo_avaliado.named_steps["classificador"]).__name__)
    print("Melhor acuracia media na validacao cruzada:", round(busca.best_score_, 4))
    print("Acuracia no conjunto de teste:", round(accuracy_score(y_teste, predicoes), 4))
    print("Melhores parametros:", busca.best_params_)
    print("Matriz de confusao (ordem das classes:", list(busca.classes_), "):")
    print(confusion_matrix(y_teste, predicoes, labels=busca.classes_))
    print("Relatorio de classificacao:")
    print(classification_report(y_teste, predicoes, zero_division=0))

    modelo_final = clone(modelo_avaliado)
    modelo_final.fit(X, y)
    artefato = {"modelo": modelo_final, "atributos": ATRIBUTOS}
    with ARQUIVO_MODELO.open("wb") as arquivo:
        pickle.dump(artefato, arquivo)
    print("Modelo final salvo em:", ARQUIVO_MODELO)


if __name__ == "__main__":
    treinar()

import os
import threading
from functools import lru_cache
import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

ARQUIVO_CSV = "Campeonato-brasileiro-estatisticas-2015-2023-completo.csv"
CLASSES = {0: "Empate", 1: "Mandante", 2: "Visitante"}
REMOVER = [
    "ID", "rodata", "data", "hora", "mandante", "visitante",
    "mandante_Placar", "visitante_Placar", "Vencedor", "resultado",
]
MODELOS_VALIDOS = ("xgboost", "random_forest", "knn")
TOP_EXPLICACAO = 4
JOGOS_RECENTES = 10

PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_STATIC = os.path.join(PASTA_BASE, "static")
ARQUIVO_CACHE = os.path.join(PASTA_BASE, "cache", "modelos.joblib")
VERSAO_MODELO = 1 

_trava = threading.Lock()

def _caminho_csv():
    for pasta in (os.getcwd(), os.path.dirname(PASTA_BASE), PASTA_BASE):
        caminho = os.path.join(pasta, ARQUIVO_CSV)
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError(
        f"Arquivo '{ARQUIVO_CSV}' não encontrado. Coloque-o na pasta do projeto, ao lado do main.py."
    )


@lru_cache(maxsize=1)
def dados():
    df = pd.read_csv(_caminho_csv())
    df["resultado"] = np.select(
        [df["Vencedor"] == df["mandante"], df["Vencedor"] == df["visitante"]], [1, 2], default=0
    )
    return df, df.drop(columns=REMOVER), df["resultado"]


@lru_cache(maxsize=1)
def _divisao():
    _, X, y = dados()
    return train_test_split(X.index, test_size=0.4, random_state=14, stratify=y)


@lru_cache(maxsize=1)
def _indices_teste():
    return frozenset(_divisao()[1])

def _assinatura():
    csv = _caminho_csv()
    return (VERSAO_MODELO, os.path.getsize(csv), int(os.path.getmtime(csv)))

def _carregar_cache():
    try:
        conteudo = joblib.load(ARQUIVO_CACHE)
        if conteudo.get("assinatura") == _assinatura():
            return conteudo
    except Exception:  
        pass
    return None


def _salvar_cache(conteudo):
    os.makedirs(os.path.dirname(ARQUIVO_CACHE), exist_ok=True)
    tmp = f"{ARQUIVO_CACHE}.{os.getpid()}.tmp"  
    joblib.dump(conteudo, tmp)
    os.replace(tmp, ARQUIVO_CACHE)


def _treinar_e_avaliar():
    _, X, y = dados()
    idx_treino, idx_teste = _divisao()
    X_treino, X_teste = X.loc[idx_treino], X.loc[idx_teste]
    y_treino, y_teste = y.loc[idx_treino], y.loc[idx_teste]

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=326, learning_rate=0.05,
        min_child_weight=7, gamma=0.8, max_depth=8, eval_metric="mlogloss", random_state=14,
    )
    xgb_model.fit(X_treino, y_treino)

    rf_model = RandomForestClassifier(
        n_estimators=100, criterion="entropy", max_depth=11, min_samples_split=2,
        min_samples_leaf=1, random_state=14,
    )
    rf_model.fit(X_treino, y_treino)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_treino_esc = scaler.fit_transform(X_treino)
    X_teste_esc = scaler.transform(X_teste)

    knn_model = KNeighborsClassifier(
        n_neighbors=38, weights="distance", algorithm="auto", p=2, metric="minkowski"
    )
    knn_model.fit(X_treino_esc, y_treino)

    nomes = [CLASSES[0], CLASSES[1], CLASSES[2]]

    def relatorio(pred):
        return classification_report(
            y_teste, pred, labels=[0, 1, 2], target_names=nomes, output_dict=True, zero_division=0
        )

    return {
        "assinatura": _assinatura(),
        "xgb": xgb_model, "rf": rf_model, "knn": knn_model, "scaler": scaler,
        "metricas": {
            "xgboost": relatorio(xgb_model.predict(X_teste)),
            "random_forest": relatorio(rf_model.predict(X_teste)),
            "knn": relatorio(knn_model.predict(X_teste_esc)),
        },
    }


def _graficos_existem():
    return all(os.path.exists(os.path.join(PASTA_STATIC, f)) for f in ("shap_bar_chart.png", "shap_beeswarm.png"))


def _gerar_graficos_shap(explainer, X_teste):
    os.makedirs(PASTA_STATIC, exist_ok=True)
    shap_values = explainer(X_teste)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_teste, plot_type="bar", show=False)
    plt.title("SHAP Bar Chart (Importância Global - XGBoost)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_STATIC, "shap_bar_chart.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    try:
        if len(shap_values.values.shape) == 3:
            shap.summary_plot(shap_values.values[:, :, 1], X_teste, show=False)  # classe 1 = Mandante
        else:
            shap.summary_plot(shap_values, X_teste, show=False)
    except Exception:
        shap.summary_plot(shap_values, X_teste, show=False)
    plt.title("SHAP Beeswarm Plot (Impacto dos Atributos - XGBoost, classe Mandante)",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_STATIC, "shap_beeswarm.png"), dpi=150)
    plt.close()


@lru_cache(maxsize=1)
def _treinar_modelos():
    df, X, _ = dados()
    idx_treino, idx_teste = _divisao()

    conteudo = _carregar_cache()
    retreinou = conteudo is None
    if retreinou:
        conteudo = _treinar_e_avaliar()
        _salvar_cache(conteudo)

    explainer = shap.TreeExplainer(conteudo["xgb"])  
    if retreinou or not _graficos_existem():
        _gerar_graficos_shap(explainer, X.loc[idx_teste])

    m = conteudo["metricas"]
    modelos = {
        "xgboost": (conteudo["xgb"], explainer, m["xgboost"]["accuracy"], None, m["xgboost"]),
        "random_forest": (conteudo["rf"], None, m["random_forest"]["accuracy"], None, m["random_forest"]),
        "knn": (conteudo["knn"], None, m["knn"]["accuracy"], conteudo["scaler"], m["knn"]),
    }
    estatisticas = {"total_jogos": len(df), "total_treino": len(idx_treino), "total_teste": len(idx_teste)}
    return modelos, X.mean(), estatisticas


def obter_dados_resultados():
    with _trava:
        modelos, _, stats = _treinar_modelos()
    return {
        "stats": stats,
        "metricas": {nome: modelos[nome][4] for nome in MODELOS_VALIDOS},
    }


def _obter_modelo(nome_modelo):
    with _trava:
        modelos, medias, _ = _treinar_modelos()
    return modelos[nome_modelo], medias


@lru_cache(maxsize=1)
def lista_partidas():
    df = dados()[0]
    teste = _indices_teste()
    return [
        [int(r.Index), f"{r.mandante} x {r.visitante} ({r.data})" + (" [teste]" if r.Index in teste else "")]
        for r in df.itertuples()
    ]


@lru_cache(maxsize=1)
def lista_times():
    df = dados()[0]
    return tuple(sorted(set(df["mandante"].unique()) | set(df["visitante"].unique())))


def _detalhe_partida(pid, previsto):
    r = dados()[0].loc[pid]
    real = CLASSES[int(r["resultado"])]
    return {
        "mandante": r["mandante"], "visitante": r["visitante"], "data": r["data"],
        "placar": f"{r['mandante_Placar']:g} x {r['visitante_Placar']:g}",
        "real": real, "acertou": real == previsto,
        "conjunto": "teste" if pid in _indices_teste() else "treino",
    }


def _fmt(v):
    return "não informado" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip(".")


def _pct(v):
    return f"{v:.1f}".replace(".", ",")


def prever(linha, tipo_modelo="xgboost"):
    (modelo, explicador, acuracia, scaler, _), medias = _obter_modelo(tipo_modelo)

    linha_proc = scaler.transform(linha) if tipo_modelo == "knn" else linha
    proba = modelo.predict_proba(linha_proc)[0]
    classe = int(proba.argmax())

    if tipo_modelo == "xgboost" and explicador is not None:
        contribuicoes = explicador(linha).values[0, :, classe]
        itens = []
        for j in np.argsort(-np.abs(contribuicoes)):
            col = linha.columns[j]
            itens.append({
                "nome": str(col).replace("_", " "), "valor": _fmt(linha.iloc[0, j]),
                "media": _fmt(medias[col]), "shap": float(contribuicoes[j]),
            })
        a_favor = [i for i in itens if i["shap"] > 0][:TOP_EXPLICACAO]
        contra = [i for i in itens if i["shap"] < 0][:TOP_EXPLICACAO]
    else:
        a_favor = []
        contra = []

    return {
        "previsto": CLASSES[classe],
        "pct_previsto": _pct(proba.max() * 100),
        "probabilidades": [
            {"classe": CLASSES[k], "pct": round(float(proba[k]) * 100, 1), "texto": _pct(proba[k] * 100), "previsto": k == classe}
            for k in (1, 0, 2)
        ],
        "a_favor": a_favor,
        "contra": contra,
        "acuracia": _pct(acuracia * 100),
        "partida": None,
        "modelo_usado": tipo_modelo.upper(),
    }


def _linha_confronto(time_m, time_v, n=JOGOS_RECENTES):
    """Monta a linha de atributos de um confronto hipotético.

    Cada equipe contribui apenas com as colunas do seu próprio lado:
      - mandante  -> colunas '*_mandante' dos seus últimos n jogos EM CASA
      - visitante -> colunas '*_visitante' dos seus últimos n jogos FORA
    O que não for coberto fica na média geral do dataset.
    """
    df, X, _ = dados()
    casa = df[df["mandante"] == time_m].tail(n)  
    fora = df[df["visitante"] == time_v].tail(n)
    if casa.empty:
        raise ValueError(f"Não há jogos como mandante de '{time_m}' no dataset.")
    if fora.empty:
        raise ValueError(f"Não há jogos como visitante de '{time_v}' no dataset.")

    linha = X.mean()  # Series nova: não altera o DataFrame em cache
    cols_m = [c for c in X.columns if c.endswith("_mandante")]
    cols_v = [c for c in X.columns if c.endswith("_visitante")]
    linha[cols_m] = casa[cols_m].mean()
    linha[cols_v] = fora[cols_v].mean()
    return pd.DataFrame([linha], columns=X.columns)


def simular(form):
    _, X, _ = dados()

    tipo_modelo = form.get("modelo", "xgboost")
    if tipo_modelo not in MODELOS_VALIDOS:
        raise ValueError("Modelo inválido.")

    tipo_simulacao = form.get("tipo_simulacao", "partida")  

    if tipo_simulacao == "personalizado":
        time_m = form.get("time_mandante", "")
        time_v = form.get("time_visitante", "")
        if not time_m or not time_v:
            raise ValueError("Selecione a equipa mandante e a equipa visitante.")
        if time_m == time_v:
            raise ValueError("A equipa mandante e a visitante devem ser diferentes.")
        if time_m not in lista_times() or time_v not in lista_times():
            raise ValueError("Equipa não encontrada no dataset.")

        resultado = prever(_linha_confronto(time_m, time_v), tipo_modelo)
        resultado["partida"] = {
            "mandante": time_m, "visitante": time_v, "data": "Confronto Personalizado",
            "placar": None, "real": None, "acertou": None,  
            "conjunto": "Personalizado",
        }
        return resultado

    if tipo_simulacao == "partida":
        try:
            pid = int(form.get("partida", ""))
            linha = X.loc[[pid]]
        except (ValueError, KeyError):
            raise ValueError("Selecione uma partida válida da lista.")
        resultado = prever(linha, tipo_modelo)
        resultado["partida"] = _detalhe_partida(pid, resultado["previsto"])
        return resultado

    raise ValueError("Tipo de simulação inválido.")
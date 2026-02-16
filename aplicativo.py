# app.py
# Dashboard de Logística: Centralizado vs Descentralizado
# SES (alpha por erro) + EOQ + ROP + Simulação (SimPy) + Plotly + (opcional) Monte Carlo

import math
import numpy as np
import pandas as pd
import simpy
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Utilidades
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)

def mae(y_true, y_pred):
    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)
    if len(y_true) <= 1:
        return float("nan")
    return float((y_true.iloc[1:] - y_pred.iloc[1:]).abs().mean())

def rmse(y_true, y_pred):
    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)
    if len(y_true) <= 1:
        return float("nan")
    return float(np.sqrt(((y_true.iloc[1:] - y_pred.iloc[1:]) ** 2).mean()))

def ses_fitted_and_forecast(series, alpha: float):
    s = pd.Series(series, dtype=float).reset_index(drop=True)
    if len(s) < 2:
        raise ValueError("Série curta demais para SES (mín. 2 pontos).")
    level = s.iloc[0]
    fitted = [level]
    for t in range(1, len(s)):
        level = alpha * s.iloc[t - 1] + (1 - alpha) * level
        fitted.append(level)
    fitted = pd.Series(fitted, dtype=float)
    forecast_next = alpha * s.iloc[-1] + (1 - alpha) * level
    return fitted, float(forecast_next)

def choose_alpha(series, alphas=None, criterio="MAE"):
    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 19)

    best = None
    for a in alphas:
        fitted, fc = ses_fitted_and_forecast(series, a)
        m = mae(series, fitted)
        r = rmse(series, fitted)
        score = m if criterio.upper() == "MAE" else r
        row = {"alpha": float(a), "MAE": float(m), "RMSE": float(r), "score": float(score),
               "fitted": fitted, "forecast": float(fc)}
        if best is None or row["score"] < best["score"]:
            best = row
    return best

def eoq(D_anual: float, S_pedido: float, H_anual: float) -> float:
    if D_anual <= 0 or S_pedido <= 0 or H_anual <= 0:
        return 1.0
    return math.sqrt((2 * D_anual * S_pedido) / H_anual)

def z_from_service_level(p: float) -> float:
    # fallback simples: aproxima níveis comuns
    tabela = {0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.95: 1.645, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}
    ch = min(tabela.keys(), key=lambda k: abs(k - p))
    return float(tabela[ch])

# =========================
# Núcleo da simulação (SimPy)
# =========================
class CentroDistribuicao:
    def __init__(self, env: simpy.Environment, nome: str, p: dict):
        self.env = env
        self.nome = nome
        self.p = p

        self.estoque = float(p["estoque_inicial"])
        self.on_order = 0.0
        self.pipeline = 0

        # séries
        self.dias = []
        self.estoque_hist = []
        self.demanda_hist = []
        self.atendida_hist = []
        self.perdida_hist = []
        self.pedidos_hist = []  # quant. pedido emitido no dia (0 ou Q)

        # custos
        self.custo_pedido = 0.0
        self.custo_frete = 0.0
        self.custo_hold = 0.0
        self.custo_rupt = 0.0

        env.process(self.run())

    def _calc_rop(self):
        # ROP = mu*L + z*sqrt(L*sigma^2 + mu^2*sigmaL^2)
        mu = self.p["demanda_media"]
        sig = self.p["demanda_std"]
        L = self.p["lead_time_media"]
        sigL = self.p["lead_time_std"]
        z = self.p["z"]
        demanda_LT = mu * L
        var = (L * (sig ** 2)) + ((mu ** 2) * (sigL ** 2))
        return demanda_LT + z * math.sqrt(max(0.0, var))

    def _place_order(self, Q):
        self.pipeline += 1
        self.on_order += Q
        self.custo_pedido += self.p["custo_fixo_pedido"]
        self.custo_frete += Q * self.p["custo_frete_unit"]

        lead = np.random.normal(self.p["lead_time_media"], self.p["lead_time_std"])
        lead = max(1, int(round(lead)))
        yield self.env.timeout(lead)

        self.estoque += Q
        self.on_order -= Q
        self.pipeline -= 1

    def run(self):
        while True:
            dia = int(self.env.now)
            self.dias.append(dia)
            self.estoque_hist.append(self.estoque)

            # Demanda do dia
            d = np.random.normal(self.p["demanda_media"], self.p["demanda_std"])
            d = max(0, int(round(d)))
            self.demanda_hist.append(d)

            vendido = min(self.estoque, d)
            perdido = d - vendido
            self.estoque -= vendido

            self.atendida_hist.append(vendido)
            self.perdida_hist.append(perdido)

            # custos
            self.custo_rupt += perdido * self.p["custo_ruptura_unit"]
            self.custo_hold += self.estoque * (self.p["custo_hold_anual"] / 365.0)

            # Política: (s, Q) com posição de estoque
            rop = self._calc_rop()
            posicao = self.estoque + self.on_order

            pedido_hoje = 0.0
            if posicao < rop and self.pipeline == 0:
                Q = self.p["Q"]
                pedido_hoje = Q
                self.env.process(self._place_order(Q))

            self.pedidos_hist.append(pedido_hoje)

            yield self.env.timeout(1)

def simular(params_por_cd: dict, dias: int, seed: int):
    set_seed(seed)
    env = simpy.Environment()
    cds = [CentroDistribuicao(env, nome, p) for nome, p in params_por_cd.items()]
    env.run(until=dias)

    rows = []
    for cd in cds:
        dem = sum(cd.demanda_hist)
        atd = sum(cd.atendida_hist)
        per = sum(cd.perdida_hist)
        fill_rate = atd / dem if dem > 0 else 1.0
        # proxy: % dias sem ruptura
        csl_proxy = 1.0 - float(np.mean(np.array(cd.perdida_hist) > 0))

        custo_total = cd.custo_pedido + cd.custo_frete + cd.custo_hold + cd.custo_rupt
        rows.append({
            "Local": cd.nome,
            "Demanda": dem,
            "Atendida": atd,
            "Perdida": per,
            "FillRate": fill_rate,
            "CSL_proxy": csl_proxy,
            "Custo_Pedido": cd.custo_pedido,
            "Custo_Frete": cd.custo_frete,
            "Custo_Holding": cd.custo_hold,
            "Custo_Ruptura": cd.custo_rupt,
            "Custo_Total": custo_total
        })

    df = pd.DataFrame(rows)
    return df, cds

def monte_carlo(params, dias, n_rep, seed0):
    res = []
    for k in range(n_rep):
        df, _ = simular(params, dias=dias, seed=seed0 + k)
        res.append({
            "Custo_Total": df["Custo_Total"].sum(),
            "FillRate": df["Atendida"].sum() / max(1, df["Demanda"].sum()),
            "Perdida": df["Perdida"].sum(),
            "Custo_Pedido": df["Custo_Pedido"].sum(),
            "Custo_Frete": df["Custo_Frete"].sum(),
            "Custo_Holding": df["Custo_Holding"].sum(),
            "Custo_Ruptura": df["Custo_Ruptura"].sum(),
        })
    return pd.DataFrame(res)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Dashboard de Logística", layout="wide")

st.title("📊 Dashboard de Logística: Centralizado vs. Descentralizado")
st.caption("Painel com SES (ajuste), EOQ, ROP e simulação (SimPy) para 365 dias (ou conforme você definir).")

# --- Sidebar: parâmetros
st.sidebar.header("⚙️ Parâmetros da Simulação")

dias = st.sidebar.slider("Horizonte (dias)", 30, 730, 365, 5)
seed = st.sidebar.number_input("Seed (reprodutibilidade)", min_value=0, max_value=10_000_000, value=42, step=1)

st.sidebar.subheader("Demanda (diária)")
criterio_alpha = st.sidebar.selectbox("Escolha do alpha (SES) por", ["MAE", "RMSE"], index=0)
sigma_mult = st.sidebar.slider("Volatilidade da demanda (multiplicador do desvio padrão)", 0.2, 3.0, 1.0, 0.05)

st.sidebar.subheader("Lead time")
lead_mu = st.sidebar.slider("Lead time médio (dias)", 1, 20, 7, 1)
lead_std = st.sidebar.slider("Atrasos no transporte (desvio padrão)", 0.0, 10.0, 2.5, 0.1)

st.sidebar.subheader("Nível de serviço / segurança")
service_level = st.sidebar.slider("Nível de serviço alvo (aprox.)", 0.80, 0.99, 0.95, 0.01)
z = z_from_service_level(service_level)
st.sidebar.write(f"Z aproximado: **{z:.3f}**")

st.sidebar.subheader("Custos")
S_PEDIDO = st.sidebar.number_input("Custo fixo por pedido (S)", value=150.0, step=10.0)
H_ANUAL  = st.sidebar.number_input("Custo anual de holding por unidade (H)", value=5.0, step=0.5)
C_RUPT   = st.sidebar.number_input("Penalidade por ruptura (por unidade)", value=20.0, step=1.0)

st.sidebar.subheader("Fretes (por unidade)")
frete_norte  = st.sidebar.number_input("Frete Norte (Desc)", value=2.50, step=0.10)
frete_centro = st.sidebar.number_input("Frete Centro (Desc)", value=2.20, step=0.10)
frete_sul    = st.sidebar.number_input("Frete Sul (Desc)", value=2.40, step=0.10)
frete_central= st.sidebar.number_input("Frete Central (Cent)", value=2.80, step=0.10)

st.sidebar.subheader("Estoques iniciais")
est_norte  = st.sidebar.number_input("Estoque inicial Norte", value=450, step=10)
est_centro = st.sidebar.number_input("Estoque inicial Centro", value=500, step=10)
est_sul    = st.sidebar.number_input("Estoque inicial Sul", value=420, step=10)
est_central= st.sidebar.number_input("Estoque inicial Central", value=1200, step=10)

st.sidebar.subheader("Monte Carlo (opcional)")
do_mc = st.sidebar.checkbox("Rodar Monte Carlo", value=False)
n_rep = st.sidebar.slider("Repetições", 20, 400, 120, 10)

# --- Dados: histórico diário por região (upload)
st.sidebar.header("📥 Dados (opcional)")
st.sidebar.caption("Envie um CSV com colunas: date, Norte, Centro, Sul (valores diários).")
up = st.sidebar.file_uploader("Upload CSV", type=["csv"])

def carregar_series_diarias():
    if up is None:
        # fallback (use seus valores reais aqui se quiser)
        # Você pode substituir por dados do seu projeto.
        # Series sintéticas só para rodar sem upload.
        base = pd.date_range("2025-01-01", periods=120, freq="D")
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            "date": base,
            "Norte":  rng.normal(3.3, 0.9, len(base)).clip(0).round().astype(int),
            "Centro": rng.normal(4.1, 1.1, len(base)).clip(0).round().astype(int),
            "Sul":    rng.normal(3.0, 0.8, len(base)).clip(0).round().astype(int),
        })
        return df
    df = pd.read_csv(up)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    # garante colunas
    for c in ["Norte", "Centro", "Sul"]:
        if c not in df.columns:
            raise ValueError(f"CSV precisa da coluna '{c}'.")
    return df

try:
    df_hist = carregar_series_diarias()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# --- SES por região
def parametros_regiao(reg_name: str, serie: pd.Series):
    best = choose_alpha(serie.values, criterio=criterio_alpha)
    mu = float(best["forecast"])  # diário
    # sigma diário: se critério for RMSE, use RMSE; senão use 1.25*MAE (regra prática)
    sigma = float(best["RMSE"]) if criterio_alpha == "RMSE" else float(1.25 * best["MAE"])
    sigma *= float(sigma_mult)
    return {
        "alpha": best["alpha"],
        "MAE": best["MAE"],
        "RMSE": best["RMSE"],
        "mu": mu,
        "sigma": sigma,
        "fitted": best["fitted"],
        "forecast": best["forecast"],
    }

reg_data = {}
for reg in ["Norte", "Centro", "Sul"]:
    reg_data[reg] = parametros_regiao(reg, df_hist[reg])

# --- EOQ/Q por CD (usa demanda anual estimada)
def Q_from_mu(mu_diario: float):
    D = mu_diario * 365.0
    return float(eoq(D, S_PEDIDO, H_ANUAL))

Q_norte  = Q_from_mu(reg_data["Norte"]["mu"])
Q_centro = Q_from_mu(reg_data["Centro"]["mu"])
Q_sul    = Q_from_mu(reg_data["Sul"]["mu"])

mu_ag = reg_data["Norte"]["mu"] + reg_data["Centro"]["mu"] + reg_data["Sul"]["mu"]
sig_ag = math.sqrt(reg_data["Norte"]["sigma"]**2 + reg_data["Centro"]["sigma"]**2 + reg_data["Sul"]["sigma"]**2)
Q_ag = Q_from_mu(mu_ag)

# --- montar params
def params_cd(mu, sigma, Q, estoque_ini, frete_unit):
    return {
        "demanda_media": float(mu),
        "demanda_std": float(sigma),
        "lead_time_media": float(lead_mu),
        "lead_time_std": float(lead_std),
        "z": float(z),
        "Q": float(Q),
        "estoque_inicial": float(estoque_ini),
        "custo_fixo_pedido": float(S_PEDIDO),
        "custo_frete_unit": float(frete_unit),
        "custo_hold_anual": float(H_ANUAL),
        "custo_ruptura_unit": float(C_RUPT),
    }

params_desc = {
    "Centro (Desc)": params_cd(reg_data["Centro"]["mu"], reg_data["Centro"]["sigma"], Q_centro, est_centro, frete_centro),
    "Norte (Desc)":  params_cd(reg_data["Norte"]["mu"],  reg_data["Norte"]["sigma"],  Q_norte,  est_norte,  frete_norte),
    "Sul (Desc)":    params_cd(reg_data["Sul"]["mu"],    reg_data["Sul"]["sigma"],    Q_sul,    est_sul,    frete_sul),
}
params_cent = {
    "Centralizado (Agregado)": params_cd(mu_ag, sig_ag, Q_ag, est_central, frete_central)
}

# --- Rodar simulações (cache leve por parâmetros principais)
@st.cache_data(show_spinner=False)
def run_once(params_desc, params_cent, dias, seed):
    df_desc, cds_desc = simular(params_desc, dias=dias, seed=seed)
    df_cent, cds_cent = simular(params_cent, dias=dias, seed=seed)
    return df_desc, cds_desc, df_cent, cds_cent

with st.spinner("Rodando simulação..."):
    df_desc, cds_desc, df_cent, cds_cent = run_once(params_desc, params_cent, dias, seed)

# =========================
# Layout principal
# =========================
colA, colB = st.columns([2, 1], gap="large")

with colB:
    st.subheader("📌 Resumo (Etapa 1)")
    resumo = pd.DataFrame([{
        "Região": r,
        "alpha": reg_data[r]["alpha"],
        "MAE": reg_data[r]["MAE"],
        "RMSE": reg_data[r]["RMSE"],
        "Forecast (mu)": reg_data[r]["mu"],
        "sigma (usada)": reg_data[r]["sigma"],
        "Q (EOQ)": Q_from_mu(reg_data[r]["mu"]),
    } for r in ["Norte", "Centro", "Sul"]])
    st.dataframe(resumo, use_container_width=True, hide_index=True)

    st.markdown("**Centralizado (agregado)**")
    st.write({
        "mu_agregado": round(mu_ag, 3),
        "sigma_agregado": round(sig_ag, 3),
        "Q_agregado": round(Q_ag, 1),
        "z": round(z, 3),
    })

with colA:
    st.subheader("1) Evolução do Estoque: Comparativo Diário")

    # gráfico único: centralizado + 3 regiões (desc)
    fig_stock = go.Figure()

    # centralizado
    ccent = cds_cent[0]
    fig_stock.add_trace(go.Scatter(
        x=ccent.dias, y=ccent.estoque_hist, mode="lines",
        name="Centralizado (Agregado)",
        hovertemplate="Dia=%{x}<br>Estoque=%{y:.0f}<extra></extra>"
    ))

    # descentralizado (3 CDs)
    # ordenar para ficar como legenda "Centro, Norte, Sul"
    for name in ["Centro (Desc)", "Norte (Desc)", "Sul (Desc)"]:
        cd = next(x for x in cds_desc if x.nome == name)
        fig_stock.add_trace(go.Scatter(
            x=cd.dias, y=cd.estoque_hist, mode="lines",
            name=name,
            hovertemplate="Dia=%{x}<br>Estoque=%{y:.0f}<extra></extra>"
        ))

    fig_stock.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Dia",
        yaxis_title="Estoque (unidades)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_stock, use_container_width=True)

st.divider()

# =========================
# Resultados financeiros + serviço
# =========================
st.subheader("2) Resultado Financeiro e Nível de Serviço")

def resumo_cenario(df):
    dem = df["Demanda"].sum()
    atd = df["Atendida"].sum()
    per = df["Perdida"].sum()
    fill = atd / max(1, dem)
    custo = df["Custo_Total"].sum()
    rupt_dias_proxy = None
    return {
        "Demanda": dem, "Atendida": atd, "Perdida": per,
        "FillRate": fill, "CustoTotal": custo
    }

r_desc = resumo_cenario(df_desc)
r_cent = resumo_cenario(df_cent)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Custo Total (Desc)", f"R$ {r_desc['CustoTotal']:,.2f}")
c2.metric("Custo Total (Cent)", f"R$ {r_cent['CustoTotal']:,.2f}")
c3.metric("Fill Rate (Desc)", f"{100*r_desc['FillRate']:.2f}%")
c4.metric("Fill Rate (Cent)", f"{100*r_cent['FillRate']:.2f}%")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.caption("Quebra de custos (Descentralizado)")
    df_break_desc = df_desc[["Local","Custo_Pedido","Custo_Frete","Custo_Holding","Custo_Ruptura","Custo_Total"]].copy()
    st.dataframe(df_break_desc, use_container_width=True, hide_index=True)

    fig_break_desc = px.bar(
        df_break_desc,
        x="Local",
        y=["Custo_Pedido","Custo_Frete","Custo_Holding","Custo_Ruptura"],
        title="Composição de custo (Desc)",
        barmode="stack"
    )
    fig_break_desc.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig_break_desc, use_container_width=True)

with col2:
    st.caption("Quebra de custos (Centralizado)")
    df_break_cent = df_cent[["Local","Custo_Pedido","Custo_Frete","Custo_Holding","Custo_Ruptura","Custo_Total"]].copy()
    st.dataframe(df_break_cent, use_container_width=True, hide_index=True)

    fig_break_cent = px.bar(
        df_break_cent,
        x="Local",
        y=["Custo_Pedido","Custo_Frete","Custo_Holding","Custo_Ruptura"],
        title="Composição de custo (Cent)",
        barmode="stack"
    )
    fig_break_cent.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig_break_cent, use_container_width=True)

st.divider()

# =========================
# Etapa 1: gráfico SES por região (todas)
# =========================
st.subheader("3) Ajuste SES (Etapa 1): Real vs Ajuste vs Forecast (por região)")

fig_ses = go.Figure()
for reg in ["Norte","Centro","Sul"]:
    serie = df_hist[reg].astype(float).reset_index(drop=True)
    fitted = reg_data[reg]["fitted"]
    fc = reg_data[reg]["forecast"]
    a = reg_data[reg]["alpha"]
    m = reg_data[reg]["MAE"]
    r = reg_data[reg]["RMSE"]

    x = list(range(1, len(serie)+1))
    fig_ses.add_trace(go.Scatter(x=x, y=serie, mode="lines", name=f"{reg} - Real"))
    fig_ses.add_trace(go.Scatter(
        x=x, y=fitted, mode="lines",
        name=f"{reg} - SES (α={a:.2f}, {criterio_alpha}={m:.2f if criterio_alpha=='MAE' else r:.2f})"
    ))
    fig_ses.add_trace(go.Scatter(x=[len(serie)+1], y=[fc], mode="markers", name=f"{reg} - Forecast"))

fig_ses.update_layout(
    height=420,
    margin=dict(l=10,r=10,t=40,b=10),
    xaxis_title="Dia (índice do histórico)",
    yaxis_title="Demanda (unid.)",
    hovermode="x unified"
)
st.plotly_chart(fig_ses, use_container_width=True)

# =========================
# Monte Carlo (opcional)
# =========================
if do_mc:
    st.divider()
    st.subheader("4) Monte Carlo (robustez): Distribuição de custo e serviço")

    with st.spinner("Rodando Monte Carlo..."):
        mc_desc = monte_carlo(params_desc, dias=dias, n_rep=n_rep, seed0=seed*100 + 1)
        mc_cent = monte_carlo(params_cent, dias=dias, n_rep=n_rep, seed0=seed*100 + 2)

    df_mc = pd.DataFrame({
        "Custo_Total": pd.concat([mc_desc["Custo_Total"], mc_cent["Custo_Total"]], ignore_index=True),
        "FillRate": pd.concat([mc_desc["FillRate"], mc_cent["FillRate"]], ignore_index=True),
        "Cenário": (["Descentralizado"] * len(mc_desc)) + (["Centralizado"] * len(mc_cent))
    })

    cL, cR = st.columns(2, gap="large")
    with cL:
        fig_cost = px.histogram(df_mc, x="Custo_Total", color="Cenário", nbins=40, barmode="overlay", marginal="box",
                                title="Distribuição do custo total (MC)")
        fig_cost.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_cost, use_container_width=True)

    with cR:
        fig_fill = px.histogram(df_mc, x="FillRate", color="Cenário", nbins=35, barmode="overlay", marginal="box",
                                title="Distribuição do Fill Rate (MC)")
        fig_fill.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_fill, use_container_width=True)

    st.caption("Dica: use a mediana e o p95 de custo para falar de risco; use o FillRate médio e p5 para serviço.")

st.success("✅ Pronto. Ajuste os sliders à esquerda e observe como custo e serviço mudam.")

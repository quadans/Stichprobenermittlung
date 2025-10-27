import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, betabinom

st.set_page_config(layout="wide")

# --- Funktionen für die Bayes-Analyse ---

def get_credible_interval(probabilities, confidence_level):
    """Berechnet das Glaubwürdigkeitsintervall aus einer Wahrscheinlichkeitsverteilung."""
    # Sortiere die Wahrscheinlichkeiten, um die wichtigsten zu finden
    sorted_indices = np.argsort(probabilities)[::-1]
    
    cumulative_prob = 0.0
    credible_set = []
    
    for i in sorted_indices:
        cumulative_prob += probabilities[i]
        credible_set.append(i)
        if cumulative_prob >= confidence_level:
            break
            
    return min(credible_set), max(credible_set)

# --- Aufbau der Streamlit App ---
st.title("Bayes'sche Analyse für Stichproben")
st.write("Untersuchen Sie, wie Stichprobendaten Ihr Wissen über die Gesamtfehlerzahl aktualisieren.")

# --- SEITENLEISTE FÜR EINGABEN ---
st.sidebar.header("Parameter festlegen")

# 1. Parameter der Stichprobe
st.sidebar.subheader("1. Stichprobendaten")
N_population = st.sidebar.number_input("Größe der Grundgesamtheit (N)", min_value=10, value=100, step=10)
n_sample = st.sidebar.number_input("Größe der Stichprobe (n)", min_value=1, value=15, step=1)
c_defects_found = st.sidebar.number_input(
    "Gefundene Fehler in der Stichprobe (c)", 
    min_value=0, max_value=n_sample, value=0, step=1
)

# 2. Definition des Priors
st.sidebar.subheader("2. Vorwissen (Prior)")
prior_type = st.sidebar.selectbox(
    "Art des Priors wählen",
    ["Uninformiert (Alle Fehlerzahlen gleich wahrscheinlich)", "Informiert (Wenige Fehler sind wahrscheinlicher)"]
)

# Mögliche Anzahl an Fehlern (Hypothesenraum K)
possible_K = np.arange(0, N_population + 1)
prior = np.zeros(N_population + 1)

if prior_type == "Uninformiert (Alle Fehlerzahlen gleich wahrscheinlich)":
    prior = np.ones(N_population + 1) / (N_population + 1)
    st.sidebar.info("Der uninformierte Prior gibt jeder möglichen Fehleranzahl von 0 bis N die exakt gleiche Startwahrscheinlichkeit.")
else:
    st.sidebar.markdown("Modellieren Sie Ihr Vorwissen mit einer Beta-Binomial-Verteilung. Verändern Sie α und β, um die Form des Priors anzupassen.")
    alpha_prior = st.sidebar.slider("Parameter α (alpha)", 0.1, 10.0, 1.0, 0.1, help="Höhere Werte zentrieren die Verteilung stärker.")
    beta_prior = st.sidebar.slider("Parameter β (beta)", 1.0, 50.0, 25.0, 1.0, help="Höhere Werte machen die Verteilung schmaler und favorisieren wenige Fehler.")
    prior = betabinom.pmf(possible_K, N_population, alpha_prior, beta_prior)
    # Sicherstellen, dass die Summe 1 ist
    prior /= np.sum(prior)

# 3. Parameter für das Ergebnis
st.sidebar.subheader("3. Ergebnisanalyse")
credibility_percent = st.sidebar.slider(
    "Glaubwürdigkeitsniveau für Intervall", 70, 99, 95, 1, format="%d%%"
)
credibility_level = credibility_percent / 100.0


# --- BERECHNUNGEN ---

# Likelihood: P(Daten | K) für jeden möglichen Wert von K
# Wahrscheinlichkeit, c Fehler in n zu finden, wenn es insgesamt K Fehler gibt
likelihood = hypergeom.pmf(c_defects_found, N_population, possible_K, n_sample)

# Posterior: P(K | Daten) ∝ P(Daten | K) * P(K)
unnormalized_posterior = likelihood * prior
# Normalisieren, damit die Summe der Wahrscheinlichkeiten 1 ergibt
posterior = unnormalized_posterior / np.sum(unnormalized_posterior)


# --- HAUPTBEREICH FÜR ERGEBNISSE ---
col1, col2 = st.columns([1.5, 2])

with col1:
    st.subheader("Ergebnisse der Analyse")
    
    # Wahrscheinlichster Wert (Maximum a Posteriori - MAP)
    map_estimate = np.argmax(posterior)
    st.metric(label="Wahrscheinlichste Fehleranzahl (MAP)", value=f"{map_estimate}")
    
    # Glaubwürdigkeitsintervall
    lower_bound, upper_bound = get_credible_interval(posterior, credibility_level)
    st.metric(
        label=f"{credibility_percent}% Glaubwürdigkeitsintervall",
        value=f"{lower_bound} – {upper_bound}"
    )
    
    st.markdown(f"""
    **Interpretation:**
    Basierend auf Ihrem Vorwissen und den Daten Ihrer Stichprobe (`n={n_sample}`, `c={c_defects_found}`):
    - Ist die wahrscheinlichste Anzahl an Fehlern in der Grundgesamtheit **{map_estimate}**.
    - Liegt die wahre Anzahl an Fehlern mit **{credibility_percent}%iger Wahrscheinlichkeit** zwischen **{lower_bound} und {upper_bound}**.
    """)


with col2:
    st.subheader("Verteilungen: Von Prior zu Posterior")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot
    ax.plot(possible_K, prior, 'b--', label='Prior (Ihr Vorwissen)', alpha=0.7)
    ax.plot(possible_K, posterior, 'r-', label='Posterior (Wissen nach der Stichprobe)', lw=2.5)
    
    # Glaubwürdigkeitsintervall im Plot hervorheben
    fill_mask = (possible_K >= lower_bound) & (possible_K <= upper_bound)
    ax.fill_between(possible_K, posterior, where=fill_mask, color='red', alpha=0.2, label=f"{credibility_percent}% Glaubwürdigkeitsintervall")
    
    ax.legend()
    ax.set_title("Aktualisierung des Wissens durch Daten", fontsize=16)
    ax.set_xlabel("Mögliche Anzahl an Fehlern (K) in der Grundgesamtheit", fontsize=12)
    ax.set_ylabel("Wahrscheinlichkeit", fontsize=12)
    ax.set_xlim(left=0, right=max(upper_bound + 10, 30)) # Zoom auf den relevanten Bereich
    
    st.pyplot(fig)
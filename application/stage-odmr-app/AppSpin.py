import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Spin 1 — Niveaux triplet")

# Matrices spin-1, base |+1>, |0>, |-1>
Sx = (1 / np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex)

Sy = (1 / np.sqrt(2)) * np.array([
    [0, -1j, 0],
    [1j, 0, -1j],
    [0, 1j, 0]
], dtype=complex)

Sz = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]
], dtype=complex)

I = np.eye(3, dtype=complex)

def get_levels(D, E, gamma, B, theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    Bx = B * np.sin(theta) * np.cos(phi)
    By = B * np.sin(theta) * np.sin(phi)
    Bz = B * np.cos(theta)

    Hzfs = D * (Sz @ Sz - (2 / 3) * I) + E * (Sx @ Sx - Sy @ Sy)
    Hzee = gamma * (Bx * Sx + By * Sy + Bz * Sz)

    H = Hzfs + Hzee

    levels = np.linalg.eigvalsh(H).real
    levels = np.sort(levels)

    return levels

st.sidebar.header("Paramètres")

D = st.sidebar.number_input("D (MHz)", value=1396.0)
E = st.sidebar.number_input("E (MHz)", value=-53.0)
gamma = st.sidebar.number_input("gamma_e (MHz/mT)", value=28.024)

B = st.sidebar.slider("B0 (mT)", 0.0, 100.0, 0.0, 0.1)
theta = st.sidebar.slider("theta (degrés)", 0.0, 180.0, 0.0, 1.0)
phi = st.sidebar.slider("phi (degrés)", 0.0, 360.0, 0.0, 1.0)

levels = get_levels(D, E, gamma, B, theta, phi)

st.subheader("Niveaux absolus")
st.write("E0, E1, E2 en MHz :")
st.write(levels)

relative = levels - levels[0]

st.subheader("Niveaux relatifs")
st.write("Niveau le plus bas fixé à 0 MHz :")
st.write(relative)

st.subheader("Transitions")
st.write(f"E1 - E0 = {relative[1] - relative[0]:.3f} MHz")
st.write(f"E2 - E1 = {relative[2] - relative[1]:.3f} MHz")
st.write(f"E2 - E0 = {relative[2] - relative[0]:.3f} MHz")

# Graphique niveaux à B fixé
fig, ax = plt.subplots(figsize=(4, 5))

for i, val in enumerate(relative):
    ax.hlines(val, -0.3, 0.3)
    ax.text(0.35, val, f"E{i} = {val:.2f} MHz", va="center")

ax.set_xlim(-1, 1.5)
ax.set_xticks([])
ax.set_ylabel("Énergie relative (MHz)")
ax.set_title(f"Niveaux à B0 = {B:.2f} mT")

st.pyplot(fig)

# Balayage en champ
st.subheader("Balayage en champ magnétique")

Bmax = st.slider("B max pour le balayage (mT)", 1.0, 300.0, 100.0)
B_values = np.linspace(0, Bmax, 300)

all_levels = []

for Bb in B_values:
    lv = get_levels(D, E, gamma, Bb, theta, phi)
    lv = lv - lv[0]
    all_levels.append(lv)

all_levels = np.array(all_levels)

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(B_values, all_levels[:, 0], label="E0")
ax2.plot(B_values, all_levels[:, 1], label="E1")
ax2.plot(B_values, all_levels[:, 2], label="E2")

ax2.set_xlabel("B0 (mT)")
ax2.set_ylabel("Énergie relative (MHz)")
ax2.set_title("Niveaux en fonction de B0")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# Transitions
f01 = all_levels[:, 1] - all_levels[:, 0]
f12 = all_levels[:, 2] - all_levels[:, 1]
f02 = all_levels[:, 2] - all_levels[:, 0]

fig3, ax3 = plt.subplots(figsize=(7, 5))

ax3.plot(B_values, f01, label="E1 - E0")
ax3.plot(B_values, f12, label="E2 - E1")
ax3.plot(B_values, f02, label="E2 - E0")

ax3.set_xlabel("B0 (mT)")
ax3.set_ylabel("Fréquence de transition (MHz)")
ax3.set_title("Transitions en fonction de B0")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)
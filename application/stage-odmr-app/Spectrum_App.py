import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks
from streamlit_plotly_events import plotly_events


st.set_page_config(
    page_title="Éditeur interactif de spectres",
    page_icon="📈",
    layout="wide",
)

NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[Ee][+-]?\d+)?"
)

DEFAULT_COLORS = [
    "#000000", "#D62728", "#1F77B4", "#2CA02C", "#FF7F0E",
    "#9467BD", "#8C564B", "#E377C2", "#17BECF",
]


# ============================================================
# LECTURE ET TRAITEMENT
# ============================================================

def read_spectrum_file(uploaded_file) -> pd.DataFrame:
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    has_marker = any("Begin Spectral Data" in line for line in lines)
    start_reading = not has_marker

    x_values, y_values = [], []

    for line in lines:
        if "Begin Spectral Data" in line:
            start_reading = True
            continue

        if not start_reading:
            continue

        values = NUMBER_PATTERN.findall(line.strip())
        if len(values) < 2:
            continue

        try:
            x = float(values[0].replace(",", "."))
            y = float(values[1].replace(",", "."))
        except ValueError:
            continue

        if np.isfinite(x) and np.isfinite(y):
            x_values.append(x)
            y_values.append(y)

    if not x_values:
        raise ValueError("Aucune paire de valeurs numériques trouvée.")

    return (
        pd.DataFrame({"x": x_values, "y": y_values})
        .dropna()
        .sort_values("x")
        .reset_index(drop=True)
    )


def find_spectrum_peaks(
    data: pd.DataFrame,
    prominence: float,
    distance: int,
    max_peaks: int,
    minima: bool,
) -> pd.DataFrame:
    x = data["x"].to_numpy(float)
    y = data["y"].to_numpy(float)

    indices, properties = find_peaks(
        -y if minima else y,
        prominence=float(prominence),
        distance=max(1, int(distance)),
    )

    return (
        pd.DataFrame(
            {
                "x": x[indices],
                "y": y[indices],
                "prominence": properties["prominences"],
            }
        )
        .sort_values("prominence", ascending=False)
        .head(int(max_peaks))
        .sort_values("x")
        .reset_index(drop=True)
    )


def parse_number_list(text: str):
    if not text.strip():
        return None

    separator = ";" if ";" in text else ","
    return [
        float(value.strip().replace(",", "."))
        for value in text.split(separator)
        if value.strip()
    ]


def axis_tick_controls(prefix: str, title: str):
    st.markdown(f"#### {title}")

    mode = st.selectbox(
        "Ticks",
        ["Automatiques", "Espacement régulier", "Positions personnalisées"],
        key=f"{prefix}_tick_mode",
    )

    settings = {
        "mode": mode,
        "tick0": None,
        "dtick": None,
        "tickvals": None,
        "ticktext": None,
    }

    if mode == "Espacement régulier":
        a, b = st.columns(2)
        settings["tick0"] = a.number_input(
            "Premier tick",
            value=0.0,
            format="%.8g",
            key=f"{prefix}_tick0",
        )
        settings["dtick"] = b.number_input(
            "Intervalle",
            min_value=1e-15,
            value=1.0,
            format="%.8g",
            key=f"{prefix}_dtick",
        )

    elif mode == "Positions personnalisées":
        values = st.text_input(
            "Positions",
            placeholder="400; 500; 600; 700",
            key=f"{prefix}_tick_values",
        )
        labels = st.text_input(
            "Étiquettes facultatives",
            placeholder="400 nm; 500 nm; 600 nm; 700 nm",
            key=f"{prefix}_tick_labels",
        )

        try:
            settings["tickvals"] = parse_number_list(values)
        except ValueError:
            st.warning("Certaines positions de ticks ne sont pas numériques.")

        if labels.strip():
            separator = ";" if ";" in labels else ","
            settings["ticktext"] = [
                value.strip() for value in labels.split(separator)
            ]

    settings["minor"] = st.checkbox(
        "Ticks mineurs",
        value=False,
        key=f"{prefix}_minor",
    )
    settings["inside"] = st.checkbox(
        "Ticks vers l'intérieur",
        value=False,
        key=f"{prefix}_inside",
    )

    return settings


def apply_tick_settings(axis_config: dict, tick_settings: dict):
    axis_config["ticks"] = "inside" if tick_settings["inside"] else "outside"

    if tick_settings["mode"] == "Espacement régulier":
        axis_config.update(
            tickmode="linear",
            tick0=tick_settings["tick0"],
            dtick=tick_settings["dtick"],
        )

    elif (
        tick_settings["mode"] == "Positions personnalisées"
        and tick_settings["tickvals"]
    ):
        axis_config.update(
            tickmode="array",
            tickvals=tick_settings["tickvals"],
        )
        if (
            tick_settings["ticktext"]
            and len(tick_settings["ticktext"])
            == len(tick_settings["tickvals"])
        ):
            axis_config["ticktext"] = tick_settings["ticktext"]

    if tick_settings["minor"]:
        axis_config["minor"] = {
            "ticks": "inside" if tick_settings["inside"] else "outside",
            "ticklen": 3,
            "showgrid": False,
        }


# ============================================================
# ÉTAT
# ============================================================

if "selected_series" not in st.session_state:
    st.session_state.selected_series = 0


# ============================================================
# CHARGEMENT
# ============================================================

st.title("Éditeur interactif de spectres")

st.markdown(
    """
    **Interaction directe :** fais glisser le titre, la légende et les noms
    de pics. Clique sur une courbe pour la sélectionner, puis modifie son
    apparence dans le panneau placé juste sous le graphique.
    """
)

uploaded_files = st.file_uploader(
    "Fichiers de spectres",
    type=["txt", "csv", "dat", "asc"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Charge au moins un spectre.")
    st.stop()


loaded = []

for i, uploaded_file in enumerate(uploaded_files):
    try:
        data = read_spectrum_file(uploaded_file)
        loaded.append((uploaded_file, data))
    except Exception as exc:
        st.error(f"{uploaded_file.name} : {exc}")

if not loaded:
    st.stop()


# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

with st.sidebar:
    st.header("Figure")

    graph_title = st.text_input("Titre", "Spectre")
    title_x = st.slider("Titre — position X", 0.0, 1.0, 0.5, 0.01)
    title_y = st.slider("Titre — position Y", 0.80, 1.20, 1.06, 0.01)
    title_size = st.number_input("Titre — taille", 8, 60, 24)

    x_title = st.text_input("Titre X", "Wavelength (nm)")
    y_left_title = st.text_input("Titre Y gauche", "Signal")

    use_right_axis = st.checkbox("Axe Y droit", False)
    y_right_title = (
        st.text_input("Titre Y droit", "Signal droit")
        if use_right_axis else ""
    )

    show_legend = st.checkbox("Afficher la légende", True)
    legend_orientation = st.selectbox(
        "Orientation de la légende",
        ["Verticale", "Horizontale"],
        disabled=not show_legend,
    )
    legend_x = st.number_input(
        "Légende — position X",
        value=1.02,
        step=0.05,
        disabled=not show_legend,
    )
    legend_y = st.number_input(
        "Légende — position Y",
        value=1.0,
        step=0.05,
        disabled=not show_legend,
    )

    st.header("Axes")

    x_scale = st.selectbox("Échelle X", ["linear", "log"])
    y_left_scale = st.selectbox("Échelle Y gauche", ["linear", "log"])
    y_right_scale = (
        st.selectbox("Échelle Y droite", ["linear", "log"])
        if use_right_axis else "linear"
    )

    x_axis_color = st.color_picker("Couleur axe X", "#000000")
    y_left_color = st.color_picker("Couleur axe Y gauche", "#000000")
    y_right_color = (
        st.color_picker("Couleur axe Y droit", "#000000")
        if use_right_axis else "#000000"
    )

    axis_line_width = st.number_input(
        "Épaisseur des axes", 0.5, 8.0, 1.5, 0.1
    )
    tick_length = st.number_input(
        "Longueur des ticks", 0.0, 30.0, 6.0, 0.5
    )
    tick_width = st.number_input(
        "Épaisseur des ticks", 0.5, 8.0, 1.0, 0.1
    )
    tick_font_size = st.number_input(
        "Taille des nombres", 6, 40, 17
    )
    axis_title_size = st.number_input(
        "Taille des titres d'axes", 8, 50, 21
    )

    show_grid = st.checkbox("Afficher la grille", False)
    grid_color = st.color_picker("Couleur de grille", "#D9D9D9")

    plot_bg = st.color_picker("Fond du graphique", "#FFFFFF")
    paper_bg = st.color_picker("Fond extérieur", "#FFFFFF")

    st.header("Dimensions")
    figure_height = st.number_input("Hauteur", 400, 1500, 720, 20)


# ============================================================
# CONFIGURATION DES SÉRIES
# ============================================================

series = []

with st.expander("Configuration complète des séries", expanded=False):
    for i, (uploaded_file, original) in enumerate(loaded):
        st.markdown(f"### {i + 1}. {uploaded_file.name}")

        c1, c2, c3, c4 = st.columns(4)

        name = c1.text_input(
            "Nom",
            uploaded_file.name.rsplit(".", 1)[0],
            key=f"name_{i}",
        )
        axis_side = c2.selectbox(
            "Axe",
            ["gauche", "droit"] if use_right_axis else ["gauche"],
            key=f"axis_{i}",
        )
        color = c3.color_picker(
            "Couleur",
            DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            key=f"color_{i}",
        )
        dash = c4.selectbox(
            "Style",
            ["solid", "dash", "dot", "dashdot"],
            key=f"dash_{i}",
        )

        c5, c6, c7, c8 = st.columns(4)
        width = c5.number_input(
            "Épaisseur",
            0.1,
            15.0,
            2.0,
            0.1,
            key=f"width_{i}",
        )
        opacity = c6.slider(
            "Opacité",
            0.05,
            1.0,
            1.0,
            key=f"opacity_{i}",
        )
        x_mult = c7.number_input(
            "Multiplicateur X",
            value=1.0,
            format="%.8g",
            key=f"xmult_{i}",
        )
        y_mult = c8.number_input(
            "Multiplicateur Y",
            value=1.0,
            format="%.8g",
            key=f"ymult_{i}",
        )

        c9, c10 = st.columns(2)
        x_offset = c9.number_input(
            "Décalage X",
            value=0.0,
            format="%.8g",
            key=f"xoffset_{i}",
        )
        y_offset = c10.number_input(
            "Décalage Y",
            value=0.0,
            format="%.8g",
            key=f"yoffset_{i}",
        )

        data = pd.DataFrame(
            {
                "x": original["x"] * x_mult + x_offset,
                "y": original["y"] * y_mult + y_offset,
            }
        )

        show_peaks = st.checkbox(
            "Afficher les pics",
            False,
            key=f"show_peaks_{i}",
        )

        peak_options = None

        if show_peaks:
            p1, p2, p3, p4 = st.columns(4)
            minima = p1.checkbox(
                "Minima",
                False,
                key=f"minima_{i}",
            )
            prominence = p2.number_input(
                "Prominence",
                min_value=0.0,
                value=0.01,
                format="%.8g",
                key=f"prominence_{i}",
            )
            distance = p3.number_input(
                "Distance minimale",
                min_value=1,
                value=1,
                step=1,
                key=f"distance_{i}",
            )
            max_peaks = p4.number_input(
                "Nombre maximal",
                1,
                100,
                6,
                key=f"max_peaks_{i}",
            )

            q1, q2, q3, q4 = st.columns(4)
            peak_position = q1.selectbox(
                "Position initiale",
                ["Au-dessus", "En dessous", "À gauche", "À droite"],
                key=f"peak_position_{i}",
            )
            peak_xshift = q2.number_input(
                "Décalage X (px)",
                value=0,
                step=2,
                key=f"peak_xshift_{i}",
            )
            peak_yshift = q3.number_input(
                "Décalage Y (px)",
                value=16,
                step=2,
                key=f"peak_yshift_{i}",
            )
            peak_font_size = q4.number_input(
                "Taille du texte",
                6,
                40,
                11,
                key=f"peak_font_size_{i}",
            )

            peak_format = st.text_input(
                "Format",
                "{x:.0f} nm",
                key=f"peak_format_{i}",
            )

            peak_options = {
                "minima": minima,
                "prominence": prominence,
                "distance": distance,
                "max_peaks": max_peaks,
                "position": peak_position,
                "xshift": peak_xshift,
                "yshift": peak_yshift,
                "font_size": peak_font_size,
                "format": peak_format,
            }

        series.append(
            {
                "name": name,
                "axis": axis_side,
                "data": data,
                "color": color,
                "dash": dash,
                "width": width,
                "opacity": opacity,
                "peaks": peak_options,
            }
        )

        st.divider()


# ============================================================
# TICKS ET LIMITES
# ============================================================

with st.expander("Ticks et limites des axes", expanded=False):
    x_col, yl_col, yr_col = st.columns(3)

    all_x = np.concatenate([s["data"]["x"].to_numpy() for s in series])
    all_yl = np.concatenate([
        s["data"]["y"].to_numpy()
        for s in series if s["axis"] == "gauche"
    ]) if any(s["axis"] == "gauche" for s in series) else np.array([0, 1])

    with x_col:
        auto_x = st.checkbox("X automatique", True)
        xmin = st.number_input(
            "X min",
            value=float(np.nanmin(all_x)),
            disabled=auto_x,
            format="%.8g",
        )
        xmax = st.number_input(
            "X max",
            value=float(np.nanmax(all_x)),
            disabled=auto_x,
            format="%.8g",
        )
        x_ticks = axis_tick_controls("x", "Ticks X")

    with yl_col:
        auto_yl = st.checkbox("Y gauche automatique", True)
        ylmin = st.number_input(
            "Y gauche min",
            value=float(np.nanmin(all_yl)),
            disabled=auto_yl,
            format="%.8g",
        )
        ylmax = st.number_input(
            "Y gauche max",
            value=float(np.nanmax(all_yl)),
            disabled=auto_yl,
            format="%.8g",
        )
        yl_ticks = axis_tick_controls("yl", "Ticks Y gauche")

    with yr_col:
        if use_right_axis:
            right_arrays = [
                s["data"]["y"].to_numpy()
                for s in series if s["axis"] == "droit"
            ]
            all_yr = (
                np.concatenate(right_arrays)
                if right_arrays else np.array([0, 1])
            )
            auto_yr = st.checkbox("Y droit automatique", True)
            yrmin = st.number_input(
                "Y droit min",
                value=float(np.nanmin(all_yr)),
                disabled=auto_yr,
                format="%.8g",
            )
            yrmax = st.number_input(
                "Y droit max",
                value=float(np.nanmax(all_yr)),
                disabled=auto_yr,
                format="%.8g",
            )
            yr_ticks = axis_tick_controls("yr", "Ticks Y droit")
        else:
            auto_yr = True
            yrmin, yrmax = 0.0, 1.0
            yr_ticks = {
                "mode": "Automatiques",
                "tick0": None,
                "dtick": None,
                "tickvals": None,
                "ticktext": None,
                "minor": False,
                "inside": False,
            }


# ============================================================
# CRÉATION DE LA FIGURE
# ============================================================

fig = go.Figure()
annotations = []
trace_to_series = {}
peak_exports = []
data_exports = []

for i, s in enumerate(series):
    yaxis_ref = "y2" if s["axis"] == "droit" else "y"
    trace_index = len(fig.data)
    trace_to_series[trace_index] = i

    fig.add_trace(
        go.Scatter(
            x=s["data"]["x"],
            y=s["data"]["y"],
            mode="lines",
            name=s["name"],
            yaxis=yaxis_ref,
            line={
                "color": s["color"],
                "width": s["width"],
                "dash": s["dash"],
            },
            opacity=s["opacity"],
            hovertemplate=(
                f"<b>{s['name']}</b><br>"
                "x=%{x:.6g}<br>y=%{y:.6g}<extra></extra>"
            ),
        )
    )

    export_data = s["data"].copy()
    export_data.insert(0, "serie", s["name"])
    export_data.insert(1, "axe", s["axis"])
    data_exports.append(export_data)

    if s["peaks"] is not None:
        peaks = find_spectrum_peaks(
            s["data"],
            s["peaks"]["prominence"],
            s["peaks"]["distance"],
            s["peaks"]["max_peaks"],
            s["peaks"]["minima"],
        )

        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["x"],
                    y=peaks["y"],
                    mode="markers",
                    yaxis=yaxis_ref,
                    showlegend=False,
                    marker={
                        "color": s["color"],
                        "size": 8,
                        "symbol": (
                            "triangle-down"
                            if s["peaks"]["minima"]
                            else "triangle-up"
                        ),
                    },
                    hovertemplate=(
                        "x=%{x:.6g}<br>y=%{y:.6g}<extra></extra>"
                    ),
                )
            )

            export_peaks = peaks.copy()
            export_peaks.insert(0, "serie", s["name"])
            peak_exports.append(export_peaks)

            position_offsets = {
                "Au-dessus": (0, -abs(s["peaks"]["yshift"])),
                "En dessous": (0, abs(s["peaks"]["yshift"])),
                "À gauche": (-abs(s["peaks"]["xshift"] or 20), 0),
                "À droite": (abs(s["peaks"]["xshift"] or 20), 0),
            }
            ax, ay = position_offsets[s["peaks"]["position"]]

            for _, peak in peaks.iterrows():
                try:
                    label = s["peaks"]["format"].format(
                        x=peak["x"],
                        y=peak["y"],
                        prominence=peak["prominence"],
                    )
                except Exception:
                    label = f"{peak['x']:.4g}"

                annotations.append(
                    {
                        "x": peak["x"],
                        "y": peak["y"],
                        "xref": "x",
                        "yref": yaxis_ref,
                        "text": label,
                        "showarrow": True,
                        "arrowhead": 2,
                        "ax": ax,
                        "ay": ay,
                        "font": {
                            "size": s["peaks"]["font_size"],
                            "color": s["color"],
                        },
                        "arrowcolor": s["color"],
                        "bgcolor": "rgba(255,255,255,0.75)",
                        "borderpad": 2,
                    }
                )

# Le titre est une annotation : il peut être déplacé à la souris.
annotations.append(
    {
        "xref": "paper",
        "yref": "paper",
        "x": title_x,
        "y": title_y,
        "text": f"<b>{graph_title}</b>",
        "showarrow": False,
        "font": {"size": title_size, "color": "#000000"},
        "xanchor": "center",
        "yanchor": "middle",
    }
)

xaxis = {
    "title": {
        "text": x_title,
        "font": {"size": axis_title_size, "color": x_axis_color},
    },
    "type": x_scale,
    "color": x_axis_color,
    "linecolor": x_axis_color,
    "linewidth": axis_line_width,
    "ticklen": tick_length,
    "tickwidth": tick_width,
    "tickfont": {"size": tick_font_size, "color": x_axis_color},
    "showline": True,
    "mirror": True,
    "showgrid": show_grid,
    "gridcolor": grid_color,
    "zeroline": False,
    "automargin": True,
}
apply_tick_settings(xaxis, x_ticks)
if not auto_x:
    xaxis["range"] = [xmin, xmax]

yaxis = {
    "title": {
        "text": y_left_title,
        "font": {"size": axis_title_size, "color": y_left_color},
    },
    "type": y_left_scale,
    "color": y_left_color,
    "linecolor": y_left_color,
    "linewidth": axis_line_width,
    "ticklen": tick_length,
    "tickwidth": tick_width,
    "tickfont": {"size": tick_font_size, "color": y_left_color},
    "showline": True,
    "mirror": True,
    "showgrid": show_grid,
    "gridcolor": grid_color,
    "zeroline": False,
    "automargin": True,
}
apply_tick_settings(yaxis, yl_ticks)
if not auto_yl:
    yaxis["range"] = [ylmin, ylmax]

layout = {
    "xaxis": xaxis,
    "yaxis": yaxis,
    "annotations": annotations,
    "showlegend": show_legend,
    "plot_bgcolor": plot_bg,
    "paper_bgcolor": paper_bg,
    "height": figure_height,
    "hovermode": "closest",
    "dragmode": "zoom",
    "margin": {"l": 90, "r": 110 if use_right_axis else 60, "t": 100, "b": 85},
}

if use_right_axis:
    yaxis2 = {
        "title": {
            "text": y_right_title,
            "font": {"size": axis_title_size, "color": y_right_color},
        },
        "type": y_right_scale,
        "color": y_right_color,
        "linecolor": y_right_color,
        "linewidth": axis_line_width,
        "ticklen": tick_length,
        "tickwidth": tick_width,
        "tickfont": {"size": tick_font_size, "color": y_right_color},
        "showline": True,
        "showgrid": False,
        "zeroline": False,
        "overlaying": "y",
        "side": "right",
        "automargin": True,
    }
    apply_tick_settings(yaxis2, yr_ticks)
    if not auto_yr:
        yaxis2["range"] = [yrmin, yrmax]
    layout["yaxis2"] = yaxis2

if show_legend:
    layout["legend"] = {
        "x": legend_x,
        "y": legend_y,
        "orientation": (
            "h" if legend_orientation == "Horizontale" else "v"
        ),
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": "#666666",
        "borderwidth": 1,
    }

fig.update_layout(**layout)


# ============================================================
# GRAPHIQUE ET SÉLECTION PAR CLIC
# ============================================================

st.header("Graphique")

clicked = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=figure_height,
    override_width="100%",
    key="editable_spectrum_graph",
)

if clicked:
    curve_number = clicked[0].get("curveNumber")
    if curve_number in trace_to_series:
        st.session_state.selected_series = trace_to_series[curve_number]

selected_index = min(
    st.session_state.selected_series,
    len(series) - 1,
)
selected = series[selected_index]

st.caption(
    "Le titre, les noms de pics et la légende peuvent être déplacés lorsque "
    "le mode éditable de Plotly est actif. Clique sur une courbe pour la "
    "sélectionner. Les modifications par glisser-déposer restent visuelles "
    "jusqu'au prochain recalcul de Streamlit."
)

# Deuxième rendu éditable : le composant d'événements détecte le clic,
# tandis que le rendu Streamlit natif permet les déplacements.
with st.expander("Mode édition directe", expanded=True):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "editable": True,
            "scrollZoom": True,
            "displaylogo": False,
            "edits": {
                "annotationPosition": True,
                "legendPosition": True,
                "titleText": True,
                "axisTitleText": True,
            },
            "modeBarButtonsToAdd": [
                "drawline",
                "drawrect",
                "eraseshape",
            ],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "spectre_edite",
                "scale": 2,
            },
        },
        key="direct_edit_graph",
    )


# ============================================================
# PANNEAU CONTEXTUEL
# ============================================================

st.header("Élément sélectionné")

left, right = st.columns([1, 2])

with left:
    selected_name = st.selectbox(
        "Courbe active",
        options=list(range(len(series))),
        index=selected_index,
        format_func=lambda idx: series[idx]["name"],
    )
    st.session_state.selected_series = selected_name

with right:
    st.markdown(
        f"**{series[selected_name]['name']}** — utilise la section "
        "« Configuration complète des séries » pour changer sa couleur, "
        "son épaisseur, son style ou son axe. Le clic sur la courbe sert à "
        "la retrouver rapidement."
    )

st.warning(
    "Limite technique : Plotly permet de déplacer directement le titre, la "
    "légende et les annotations, mais il ne permet pas de modifier un tick "
    "ou la couleur d'une courbe simplement en cliquant sur son texte. Ces "
    "réglages restent disponibles dans les panneaux de l'application."
)


# ============================================================
# EXPORT
# ============================================================

st.header("Exports")

data_df = pd.concat(data_exports, ignore_index=True)
peaks_df = (
    pd.concat(peak_exports, ignore_index=True)
    if peak_exports
    else pd.DataFrame(columns=["serie", "x", "y", "prominence"])
)

html_bytes = fig.to_html(
    include_plotlyjs="cdn",
    full_html=True,
    config={"editable": True, "scrollZoom": True},
).encode("utf-8")

data_csv = data_df.to_csv(index=False).encode("utf-8")
peaks_csv = peaks_df.to_csv(index=False).encode("utf-8")

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("graphique_interactif.html", html_bytes)
    archive.writestr("donnees_tracees.csv", data_csv)
    archive.writestr("pics_detectes.csv", peaks_csv)

e1, e2, e3, e4 = st.columns(4)

e1.download_button(
    "Graphique HTML",
    html_bytes,
    "graphique_interactif.html",
    "text/html",
)
e2.download_button(
    "Données CSV",
    data_csv,
    "donnees_tracees.csv",
    "text/csv",
)
e3.download_button(
    "Pics CSV",
    peaks_csv,
    "pics_detectes.csv",
    "text/csv",
)
e4.download_button(
    "Tout télécharger",
    zip_buffer.getvalue(),
    "resultats_spectres.zip",
    "application/zip",
)
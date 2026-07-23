import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks


st.set_page_config(
    page_title="Traceur de spectres interactif",
    page_icon="📈",
    layout="wide",
)

NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[Ee][+-]?\d+)?"
)


# ============================================================
# LECTURE DES FICHIERS
# ============================================================

def read_spectrum_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    has_begin_marker = any("Begin Spectral Data" in line for line in lines)
    start_reading = not has_begin_marker

    x_values, y_values = [], []

    for line in lines:
        if "Begin Spectral Data" in line:
            start_reading = True
            continue

        if not start_reading:
            continue

        numbers = NUMBER_PATTERN.findall(line.strip())

        if len(numbers) < 2:
            continue

        try:
            x = float(numbers[0].replace(",", "."))
            y = float(numbers[1].replace(",", "."))
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


def detect_peaks(
    data: pd.DataFrame,
    prominence: float,
    distance: int,
    max_peaks: int,
    detect_minima: bool,
    xmin=None,
    xmax=None,
) -> pd.DataFrame:
    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)

    detection_signal = -y if detect_minima else y

    indices, properties = find_peaks(
        detection_signal,
        prominence=float(prominence),
        distance=max(1, int(distance)),
    )

    peaks = pd.DataFrame(
        {
            "x": x[indices],
            "y": y[indices],
            "prominence": properties["prominences"],
        }
    )

    if xmin is not None:
        peaks = peaks[peaks["x"] >= xmin]
    if xmax is not None:
        peaks = peaks[peaks["x"] <= xmax]

    return (
        peaks.sort_values("prominence", ascending=False)
        .head(int(max_peaks))
        .sort_values("x")
        .reset_index(drop=True)
    )


def axis_type(scale: str) -> str:
    # Plotly ne possède pas une échelle symlog native.
    return "log" if scale == "log" else "linear"


def tick_mode_controls(prefix: str, axis_name: str):
    st.markdown(f"##### Graduations — {axis_name}")

    mode = st.selectbox(
        "Mode des ticks",
        ["Automatique", "Pas régulier", "Positions personnalisées"],
        key=f"{prefix}_tick_mode",
    )

    tick0 = None
    dtick = None
    tickvals = None
    ticktext = None

    if mode == "Pas régulier":
        c1, c2 = st.columns(2)
        with c1:
            tick0 = st.number_input(
                "Premier tick",
                value=0.0,
                format="%.8g",
                key=f"{prefix}_tick0",
            )
        with c2:
            dtick = st.number_input(
                "Espacement entre les ticks",
                min_value=1e-15,
                value=1.0,
                format="%.8g",
                key=f"{prefix}_dtick",
            )

    elif mode == "Positions personnalisées":
        values_text = st.text_input(
            "Positions, séparées par des virgules",
            value="",
            placeholder="Ex. 400, 500, 600, 700",
            key=f"{prefix}_tickvals",
        )
        labels_text = st.text_input(
            "Étiquettes facultatives, séparées par des virgules",
            value="",
            placeholder="Ex. 400 nm, 500 nm, 600 nm, 700 nm",
            key=f"{prefix}_ticktext",
        )

        if values_text.strip():
            try:
                tickvals = [
                    float(v.strip().replace(",", "."))
                    for v in values_text.split(";")
                ] if ";" in values_text else [
                    float(v.strip())
                    for v in values_text.split(",")
                ]
            except ValueError:
                st.warning(
                    f"Les positions personnalisées de {axis_name} sont invalides."
                )
                tickvals = None

        if labels_text.strip():
            separator = ";" if ";" in labels_text else ","
            ticktext = [v.strip() for v in labels_text.split(separator)]

            if tickvals is not None and len(ticktext) != len(tickvals):
                st.warning(
                    f"Le nombre d'étiquettes de {axis_name} doit correspondre "
                    "au nombre de positions."
                )
                ticktext = None

    minor_ticks = st.checkbox(
        "Afficher les ticks mineurs",
        value=False,
        key=f"{prefix}_minor",
    )

    ticks_inside = st.checkbox(
        "Ticks vers l'intérieur",
        value=False,
        key=f"{prefix}_inside",
    )

    return {
        "mode": mode,
        "tick0": tick0,
        "dtick": dtick,
        "tickvals": tickvals,
        "ticktext": ticktext,
        "minor_ticks": minor_ticks,
        "ticks": "inside" if ticks_inside else "outside",
    }


def make_axis_config(
    title,
    scale,
    color,
    title_color,
    line_width,
    tick_length,
    tick_width,
    tick_font_size,
    title_font_size,
    limits_auto,
    minimum,
    maximum,
    tick_config,
    show_grid,
    grid_color,
):
    config = {
        "title": {
            "text": title,
            "font": {"color": title_color, "size": title_font_size},
        },
        "type": axis_type(scale),
        "color": color,
        "linecolor": color,
        "linewidth": line_width,
        "mirror": True,
        "showline": True,
        "ticks": tick_config["ticks"],
        "ticklen": tick_length,
        "tickwidth": tick_width,
        "tickcolor": color,
        "tickfont": {"color": color, "size": tick_font_size},
        "showgrid": show_grid,
        "gridcolor": grid_color,
        "zeroline": False,
        "automargin": True,
    }

    if not limits_auto:
        config["range"] = [minimum, maximum]

    if tick_config["mode"] == "Pas régulier":
        config["tickmode"] = "linear"
        config["tick0"] = tick_config["tick0"]
        config["dtick"] = tick_config["dtick"]

    elif tick_config["mode"] == "Positions personnalisées":
        if tick_config["tickvals"] is not None:
            config["tickmode"] = "array"
            config["tickvals"] = tick_config["tickvals"]
            if tick_config["ticktext"] is not None:
                config["ticktext"] = tick_config["ticktext"]

    # Plotly accepte minor sur les versions récentes.
    if tick_config["minor_ticks"]:
        config["minor"] = {
            "ticks": tick_config["ticks"],
            "ticklen": max(2, tick_length * 0.6),
            "tickcolor": color,
            "showgrid": False,
        }

    return config


# ============================================================
# INTERFACE
# ============================================================

st.title("Traceur de spectres interactif")

st.info(
    "Dans le graphique, utilise la barre d'outils pour zoomer et déplacer la "
    "vue. Le mode éditable permet aussi de déplacer la légende et les "
    "annotations des pics directement à la souris. Les positions déplacées "
    "manuellement dans le graphique ne sont toutefois pas réinjectées "
    "automatiquement dans les contrôles Streamlit."
)

uploaded_files = st.file_uploader(
    "Fichiers de spectres",
    type=["txt", "csv", "dat", "asc"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()


# ============================================================
# APPARENCE GÉNÉRALE
# ============================================================

with st.sidebar:
    st.header("Graphique")

    title = st.text_input("Titre", "")
    x_title = st.text_input("Titre de l'axe X", "Wavelength (nm)")
    y_left_title = st.text_input("Titre de l'axe Y gauche", "Signal")

    use_right_axis = st.checkbox("Ajouter un axe Y droit", False)
    y_right_title = (
        st.text_input("Titre de l'axe Y droit", "Signal droit")
        if use_right_axis
        else ""
    )

    show_legend = st.checkbox("Afficher la légende", True)
    editable_graph = st.checkbox(
        "Autoriser l'édition directe du graphique",
        True,
        help=(
            "Permet notamment de déplacer les annotations et la légende "
            "directement avec la souris."
        ),
    )

    legend_orientation = st.selectbox(
        "Orientation de la légende",
        ["verticale", "horizontale"],
        disabled=not show_legend,
    )

    legend_x = st.number_input(
        "Position X de la légende",
        value=1.02,
        step=0.05,
        disabled=not show_legend,
    )
    legend_y = st.number_input(
        "Position Y de la légende",
        value=1.0,
        step=0.05,
        disabled=not show_legend,
    )

    legend_bg = st.color_picker(
        "Fond de la légende",
        "#FFFFFF",
        disabled=not show_legend,
    )
    legend_font_color = st.color_picker(
        "Texte de la légende",
        "#000000",
        disabled=not show_legend,
    )

    plot_bg = st.color_picker("Fond du graphique", "#FFFFFF")
    paper_bg = st.color_picker("Fond extérieur", "#FFFFFF")
    grid_color = st.color_picker("Couleur de la grille", "#D9D9D9")
    show_grid = st.checkbox("Afficher la grille", False)

    width = st.number_input("Largeur", 500, 2400, 1200, 50)
    height = st.number_input("Hauteur", 350, 1600, 700, 50)

    title_font_size = st.number_input("Taille du titre", 8, 60, 24)
    axis_title_size = st.number_input(
        "Taille des titres d'axes", 8, 60, 22
    )
    tick_font_size = st.number_input(
        "Taille des nombres des axes", 6, 50, 18
    )

    st.header("Axes")

    x_scale = st.selectbox("Échelle X", ["linear", "log"])
    y_left_scale = st.selectbox("Échelle Y gauche", ["linear", "log"])
    y_right_scale = (
        st.selectbox("Échelle Y droite", ["linear", "log"])
        if use_right_axis
        else "linear"
    )

    x_color = st.color_picker("Couleur de l'axe X", "#000000")
    y_left_color = st.color_picker("Couleur de l'axe Y gauche", "#000000")
    y_right_color = (
        st.color_picker("Couleur de l'axe Y droit", "#000000")
        if use_right_axis
        else "#000000"
    )

    x_title_color = st.color_picker("Couleur du titre X", x_color)
    y_left_title_color = st.color_picker(
        "Couleur du titre Y gauche", y_left_color
    )
    y_right_title_color = (
        st.color_picker("Couleur du titre Y droit", y_right_color)
        if use_right_axis
        else "#000000"
    )

    axis_width = st.number_input(
        "Épaisseur des axes", 0.5, 10.0, 1.5, 0.1
    )
    tick_width = st.number_input(
        "Épaisseur des ticks", 0.5, 10.0, 1.0, 0.1
    )
    tick_length = st.number_input(
        "Longueur des ticks", 0.0, 30.0, 6.0, 0.5
    )


# ============================================================
# SÉRIES
# ============================================================

default_colors = [
    "#000000", "#D62728", "#1F77B4", "#2CA02C", "#FF7F0E",
    "#9467BD", "#8C564B", "#E377C2", "#17BECF",
]

series = []

st.header("Configuration des séries")

for i, uploaded_file in enumerate(uploaded_files):
    try:
        original = read_spectrum_file(uploaded_file)
    except Exception as exc:
        st.error(f"{uploaded_file.name}: {exc}")
        continue

    with st.expander(
        f"Série {i + 1} — {uploaded_file.name}",
        expanded=(i == 0),
    ):
        c1, c2, c3 = st.columns(3)

        with c1:
            name = st.text_input(
                "Nom de la série",
                uploaded_file.name.rsplit(".", 1)[0],
                key=f"name_{i}",
            )
            target_axis = st.selectbox(
                "Axe Y",
                ["gauche", "droit"] if use_right_axis else ["gauche"],
                key=f"axis_{i}",
            )

        with c2:
            color = st.color_picker(
                "Couleur",
                default_colors[i % len(default_colors)],
                key=f"color_{i}",
            )
            line_width = st.number_input(
                "Épaisseur de ligne",
                0.1,
                15.0,
                2.0,
                0.1,
                key=f"lw_{i}",
            )

        with c3:
            dash = st.selectbox(
                "Style de ligne",
                ["solid", "dash", "dot", "dashdot"],
                key=f"dash_{i}",
            )
            opacity = st.slider(
                "Opacité",
                0.05,
                1.0,
                1.0,
                key=f"opacity_{i}",
            )

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            x_mult = st.number_input(
                "Multiplicateur X",
                value=1.0,
                format="%.8g",
                key=f"xmult_{i}",
            )
        with t2:
            x_offset = st.number_input(
                "Décalage X",
                value=0.0,
                format="%.8g",
                key=f"xoff_{i}",
            )
        with t3:
            y_mult = st.number_input(
                "Multiplicateur Y",
                value=1.0,
                format="%.8g",
                key=f"ymult_{i}",
            )
        with t4:
            y_offset = st.number_input(
                "Décalage Y",
                value=0.0,
                format="%.8g",
                key=f"yoff_{i}",
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
            key=f"peaks_{i}",
        )

        peak_settings = None

        if show_peaks:
            p1, p2, p3, p4 = st.columns(4)

            with p1:
                minima = st.checkbox(
                    "Détecter les minima",
                    False,
                    key=f"minima_{i}",
                )
            with p2:
                prominence = st.number_input(
                    "Prominence",
                    min_value=0.0,
                    value=0.01,
                    format="%.8g",
                    key=f"prom_{i}",
                )
            with p3:
                distance = st.number_input(
                    "Distance minimale",
                    min_value=1,
                    value=1,
                    step=1,
                    key=f"dist_{i}",
                )
            with p4:
                max_peaks = st.number_input(
                    "Nombre maximal",
                    1,
                    100,
                    6,
                    key=f"npeaks_{i}",
                )

            a1, a2, a3, a4 = st.columns(4)

            with a1:
                annotation_position = st.selectbox(
                    "Position des noms",
                    [
                        "Au-dessus",
                        "En dessous",
                        "À gauche",
                        "À droite",
                        "Décalage personnalisé",
                    ],
                    key=f"annpos_{i}",
                )

            with a2:
                annotation_xshift = st.number_input(
                    "Décalage horizontal (px)",
                    value=0,
                    step=2,
                    disabled=annotation_position != "Décalage personnalisé",
                    key=f"annx_{i}",
                )

            with a3:
                annotation_yshift = st.number_input(
                    "Décalage vertical (px)",
                    value=15,
                    step=2,
                    disabled=annotation_position != "Décalage personnalisé",
                    key=f"anny_{i}",
                )

            with a4:
                annotation_font_size = st.number_input(
                    "Taille des noms",
                    6,
                    40,
                    11,
                    key=f"annsize_{i}",
                )

            annotation_format = st.text_input(
                "Format des noms",
                "{x:.0f} nm",
                help="Variables disponibles : {x}, {y}, {prominence}",
                key=f"annformat_{i}",
            )

            peak_settings = {
                "minima": minima,
                "prominence": prominence,
                "distance": distance,
                "max_peaks": max_peaks,
                "position": annotation_position,
                "xshift": annotation_xshift,
                "yshift": annotation_yshift,
                "font_size": annotation_font_size,
                "format": annotation_format,
            }

        series.append(
            {
                "name": name,
                "axis": target_axis,
                "data": data,
                "color": color,
                "line_width": line_width,
                "dash": dash,
                "opacity": opacity,
                "peaks": peak_settings,
            }
        )

if not series:
    st.stop()


# ============================================================
# LIMITES ET TICKS
# ============================================================

all_x = np.concatenate([s["data"]["x"].to_numpy() for s in series])
left_y = [
    s["data"]["y"].to_numpy()
    for s in series
    if s["axis"] == "gauche"
]
right_y = [
    s["data"]["y"].to_numpy()
    for s in series
    if s["axis"] == "droit"
]

st.header("Limites et graduations")

c1, c2, c3 = st.columns(3)

with c1:
    auto_x = st.checkbox("Limites X automatiques", True)
    xmin = st.number_input(
        "X minimum",
        value=float(np.nanmin(all_x)),
        format="%.8g",
        disabled=auto_x,
    )
    xmax = st.number_input(
        "X maximum",
        value=float(np.nanmax(all_x)),
        format="%.8g",
        disabled=auto_x,
    )
    x_ticks = tick_mode_controls("x", "X")

with c2:
    left_values = np.concatenate(left_y) if left_y else np.array([0.0, 1.0])
    auto_yl = st.checkbox("Limites Y gauche automatiques", True)
    ylmin = st.number_input(
        "Y gauche minimum",
        value=float(np.nanmin(left_values)),
        format="%.8g",
        disabled=auto_yl,
    )
    ylmax = st.number_input(
        "Y gauche maximum",
        value=float(np.nanmax(left_values)),
        format="%.8g",
        disabled=auto_yl,
    )
    yl_ticks = tick_mode_controls("yl", "Y gauche")

with c3:
    if use_right_axis:
        right_values = (
            np.concatenate(right_y) if right_y else np.array([0.0, 1.0])
        )
        auto_yr = st.checkbox("Limites Y droite automatiques", True)
        yrmin = st.number_input(
            "Y droite minimum",
            value=float(np.nanmin(right_values)),
            format="%.8g",
            disabled=auto_yr,
        )
        yrmax = st.number_input(
            "Y droite maximum",
            value=float(np.nanmax(right_values)),
            format="%.8g",
            disabled=auto_yr,
        )
        yr_ticks = tick_mode_controls("yr", "Y droite")
    else:
        auto_yr = True
        yrmin, yrmax = 0.0, 1.0
        yr_ticks = {
            "mode": "Automatique",
            "tick0": None,
            "dtick": None,
            "tickvals": None,
            "ticktext": None,
            "minor_ticks": False,
            "ticks": "outside",
        }


# ============================================================
# GRAPHIQUE PLOTLY
# ============================================================

fig = go.Figure()

annotations = []
peak_exports = []
data_exports = []

for s in series:
    yaxis_name = "y2" if s["axis"] == "droit" else "y"

    fig.add_trace(
        go.Scatter(
            x=s["data"]["x"],
            y=s["data"]["y"],
            mode="lines",
            name=s["name"],
            yaxis=yaxis_name,
            line={
                "color": s["color"],
                "width": s["line_width"],
                "dash": s["dash"],
            },
            opacity=s["opacity"],
            hovertemplate=(
                f"<b>{s['name']}</b><br>"
                "x=%{x:.6g}<br>"
                "y=%{y:.6g}<extra></extra>"
            ),
        )
    )

    export = s["data"].copy()
    export.insert(0, "serie", s["name"])
    export.insert(1, "axe_y", s["axis"])
    data_exports.append(export)

    if s["peaks"] is not None:
        peaks = detect_peaks(
            s["data"],
            s["peaks"]["prominence"],
            s["peaks"]["distance"],
            s["peaks"]["max_peaks"],
            s["peaks"]["minima"],
            None if auto_x else xmin,
            None if auto_x else xmax,
        )

        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["x"],
                    y=peaks["y"],
                    mode="markers",
                    name=f"{s['name']} — pics",
                    yaxis=yaxis_name,
                    marker={
                        "color": s["color"],
                        "size": 8,
                        "symbol": (
                            "triangle-down"
                            if s["peaks"]["minima"]
                            else "triangle-up"
                        ),
                    },
                    showlegend=False,
                    hovertemplate=(
                        "x=%{x:.6g}<br>"
                        "y=%{y:.6g}<extra></extra>"
                    ),
                )
            )

            peaks_export = peaks.copy()
            peaks_export.insert(0, "serie", s["name"])
            peak_exports.append(peaks_export)

            position_map = {
                "Au-dessus": (0, 18, "center", "bottom"),
                "En dessous": (0, -18, "center", "top"),
                "À gauche": (-20, 0, "right", "middle"),
                "À droite": (20, 0, "left", "middle"),
                "Décalage personnalisé": (
                    s["peaks"]["xshift"],
                    s["peaks"]["yshift"],
                    "center",
                    "bottom",
                ),
            }

            xshift, yshift, xanchor, yanchor = position_map[
                s["peaks"]["position"]
            ]

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
                        "yref": yaxis_name,
                        "text": label,
                        "showarrow": True,
                        "arrowhead": 2,
                        "arrowcolor": s["color"],
                        "font": {
                            "color": s["color"],
                            "size": s["peaks"]["font_size"],
                        },
                        "ax": xshift,
                        "ay": -yshift,
                        "xanchor": xanchor,
                        "yanchor": yanchor,
                        "bgcolor": "rgba(255,255,255,0.75)",
                        "borderpad": 2,
                    }
                )

xaxis_config = make_axis_config(
    x_title,
    x_scale,
    x_color,
    x_title_color,
    axis_width,
    tick_length,
    tick_width,
    tick_font_size,
    axis_title_size,
    auto_x,
    xmin,
    xmax,
    x_ticks,
    show_grid,
    grid_color,
)

yaxis_config = make_axis_config(
    y_left_title,
    y_left_scale,
    y_left_color,
    y_left_title_color,
    axis_width,
    tick_length,
    tick_width,
    tick_font_size,
    axis_title_size,
    auto_yl,
    ylmin,
    ylmax,
    yl_ticks,
    show_grid,
    grid_color,
)

layout = {
    "title": {
        "text": title,
        "font": {"size": title_font_size},
        "x": 0.5,
    },
    "xaxis": xaxis_config,
    "yaxis": yaxis_config,
    "width": width,
    "height": height,
    "plot_bgcolor": plot_bg,
    "paper_bgcolor": paper_bg,
    "hovermode": "closest",
    "annotations": annotations,
    "showlegend": show_legend,
    "margin": {"l": 90, "r": 100 if use_right_axis else 50, "t": 80, "b": 80},
}

if use_right_axis:
    layout["yaxis2"] = make_axis_config(
        y_right_title,
        y_right_scale,
        y_right_color,
        y_right_title_color,
        axis_width,
        tick_length,
        tick_width,
        tick_font_size,
        axis_title_size,
        auto_yr,
        yrmin,
        yrmax,
        yr_ticks,
        False,
        grid_color,
    )
    layout["yaxis2"].update(
        {
            "overlaying": "y",
            "side": "right",
        }
    )

if show_legend:
    layout["legend"] = {
        "x": legend_x,
        "y": legend_y,
        "orientation": "h" if legend_orientation == "horizontale" else "v",
        "bgcolor": legend_bg,
        "font": {"color": legend_font_color},
        "bordercolor": legend_font_color,
        "borderwidth": 1,
    }

fig.update_layout(**layout)

st.header("Graphique final")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "editable": editable_graph,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawrect",
            "eraseshape",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "spectre",
            "height": height,
            "width": width,
            "scale": 2,
        },
    },
)

st.caption(
    "Pour déplacer un nom de pic : active « Autoriser l'édition directe », "
    "puis fais glisser l'annotation. Tu peux également déplacer la légende. "
    "Le bouton appareil photo de Plotly exporte directement l'état visuel "
    "actuel du graphique."
)


# ============================================================
# EXPORTS
# ============================================================

st.header("Données et exports")

data_df = pd.concat(data_exports, ignore_index=True)
peaks_df = (
    pd.concat(peak_exports, ignore_index=True)
    if peak_exports
    else pd.DataFrame(columns=["serie", "x", "y", "prominence"])
)

if not peaks_df.empty:
    st.subheader("Pics détectés")
    st.dataframe(peaks_df, use_container_width=True)

html_bytes = fig.to_html(
    include_plotlyjs="cdn",
    full_html=True,
    config={"editable": editable_graph, "scrollZoom": True},
).encode("utf-8")

data_csv = data_df.to_csv(index=False).encode("utf-8")
peaks_csv = peaks_df.to_csv(index=False).encode("utf-8")

zip_buffer = io.BytesIO()

with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("graphique_interactif.html", html_bytes)
    archive.writestr("donnees_tracees.csv", data_csv)
    archive.writestr("pics_detectes.csv", peaks_csv)

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "Graphique interactif HTML",
        html_bytes,
        "graphique_interactif.html",
        "text/html",
    )

with d2:
    st.download_button(
        "Données tracées CSV",
        data_csv,
        "donnees_tracees.csv",
        "text/csv",
    )

with d3:
    st.download_button(
        "Pics détectés CSV",
        peaks_csv,
        "pics_detectes.csv",
        "text/csv",
    )

with d4:
    st.download_button(
        "Tout télécharger ZIP",
        zip_buffer.getvalue(),
        "resultats_spectres.zip",
        "application/zip",
    )
"""
Version 1.08
Build on Python 3.11.9 with (see requirements.txt)
Contact: wink@via.rwth-aachen.de
Change History:
- 1.08, 2024-07-03 cw: Bugfix Plot der Trajektorien
- 1.07, 2024-04-08 cw: Einheitliche Dateinamenpräfixe & PEP 8 Konformität
- 1.06, 2024-04-05 cw: Bugfix Ausgabe auf Plot der Trajektorien
- 1.05, 2024-04-02 cw: Lokalisierung
- 1.04, 2024-03-25 cw: Codeoptimierung
- 1.03, 2024-02-26 cw: Codeoptimierung
- 1.02, 2023-07-27 cw: Ordner erstellen wenn nicht vorhanden, Progress Bar für Einzelbilder
- 1.01, 2023-07-26 cw: Optimierung der Ausgaben bei wechselnden Anzahlen IF-BG
- 1.00, 2023-07-12 cw: Initialer Stand mit Dokumentation und Versionierung
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np

from progress.bar import IncrementalBar

import constants
import trajectory

# Dateityp der Ausgaben
FILETYPE = "png"
# definierte Lokalisierungen
LOC_DE = "de"
LOC_EN = "en"
LOCALE = [LOC_DE, LOC_EN]


def plot_3d_shape(data: pd.DataFrame, input, output) -> None:
    """
    Plottet den gewichteten Fahrzeitverlust als 3D-Shape über alle möglichen Balisenpositionen.

    Args:
        data: gewichteter Fahrzeitverlust über die möglichen Balisenpositionen
        input: Klasse der Input-Parameter
        output: Klasse der Berechnungsergebnisse

    Raises:
        ValueError: Lokalisierung nicht definiert

    Returns:
        none
    """

    if input.tech_locale not in LOCALE:
        raise ValueError("locale not found")
    # Daten interpolieren außerhalb des optimalen Bereichs
    data.interpolate(method="linear", axis=0, limit=10, inplace=True)
    data.interpolate(method="linear", axis=1, limit=10, inplace=True)
    # Hilfswerte
    min_value = data.min(skipna=True).min()
    max_value = data.max(skipna=True).max()
    min_plot = np.floor(min_value/10) * 10
    # Auflösung
    px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(1920*px, 1080*px))
    ax = fig.add_subplot(projection="3d")
    # Daten vorbereiten
    x = data.columns + 1
    y = data.index + 1
    Z = data[x-1][y-1]
    X, Y = np.meshgrid(x-1, y-1)
    # Projektion auf x-y-Ebene
    ax.contourf(X, Y, Z, zdir="z", offset=min_plot, alpha=0.7, levels=50,
                norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value), cmap="RdYlGn_r")
    # 3D-Ansicht
    ax.plot_surface(X, Y, Z, norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value),
                    cmap="RdYlGn_r", rcount=200, ccount=200)
    # 3D-Achsenkreuz
    ax.plot([0, input.track_infill_1], [output.infill_distance_1, output.infill_distance_1],
            [min_plot, min_plot], color="black")
    ax.plot([output.infill_distance_2, output.infill_distance_2], [0, input.track_infill_1],
            [min_plot, min_plot], color="black")
    ax.plot([output.infill_distance_2, output.infill_distance_2],
            [output.infill_distance_1, output.infill_distance_1], [min_plot, min_value],
            color="black")
    # Labels
    if input.tech_locale == LOC_DE:
        ax.set_xlabel("Infill-Distanz 1 [m]")
        ax.set_ylabel("Infill-Distanz 2 [m]")
        ax.set_zlabel("gewichteter Fahrzeitverlust [s]")
    elif input.tech_locale == LOC_EN:
        ax.set_xlabel("infill distance 1 [m]")
        ax.set_ylabel("infill distance 2 [m]")
        ax.set_zlabel("weighted additional runtime [s]")

    # speichern
    path = "output"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./{path}/{input.timestr}_plot_3d.{FILETYPE}", bbox_inches="tight",
                pad_inches=0.5)
    # Einzelbilder für Animation
    if input.tech_rotate_plot:
        path = path + "/rotate"
        if not os.path.exists(path):
            os.makedirs(path)
        bar = IncrementalBar("Einzelbilder plotten", max=360, suffix="%(percent).1f%% "
                             + "abgeschlossen - Restdauer: Ungefähr %(eta)d Sekunden")
        angle_start = 300
        for angle in range(angle_start, angle_start+360, 1):
            ax.view_init(elev=30, azim=angle)
            plt.savefig(f"./{path}/plot_3d_{angle-angle_start+1:03d}.{FILETYPE}",
                        bbox_inches="tight", pad_inches=0.5)
            bar.next()
        bar.finish()


def plot_trajectory(accel_1: np.ndarray, accel_2: np.ndarray, accel_3: np.ndarray,
                    accel_4: np.ndarray, input, totals, output, factors: list) -> None:
    """
    Plottet die berechneten Trajektorien mit den unterschiedlichen Aufwertepunkten und gibt
    Inputparameter ebenso wie Ergebnisse auf dem Plot aus.

    Args:
        accel_1: Beschleunigungsstufen über den Weg der ersten Trajektorie
        accel_2: Beschleunigungsstufen über den Weg der zweiten Trajektorie
        accel_3: Beschleunigungsstufen über den Weg der dritten Trajektorie
        accel_4: Beschleunigungsstufen über den Weg der vierten Trajektorie
        input: Klasse der Input-Parameter
        totals: Klasse der verrechneten Parameter
        output: Klasse der Berechnungsergebnisse
        factors: Gewichtungsfaktoren

    Raises:
        ValueError: Lokalisierung nicht definiert
        NotImplementedError: Fehler beim Plotten, keine Fehlerbehandlung implementiert

    Returns:
        none
    """

    # definierte Lokalisierungen
    locale = input.tech_locale
    if locale not in LOCALE:
        raise ValueError("locale not found")
    # Hilfswerte
    max_speed = np.max(totals.train_speed*constants.CONVERT_MPS_KPH)
    max_distance = np.max(output.distance_infill_1)
    plot_max_speed = max_speed + 9
    plot_max_distance = np.ceil(max_distance/250) * 250 + 25
    offset = totals.track_distance_origin_target
    # Trajektorie bei Aufwertung an Infill 1 = ungehindert
    trajectory_1 = trajectory.clean(output.distance_infill_1, output.speed_infill_1, accel_1,
                                    plot_max_distance)
    # Trajektorie bei Aufwertung an Infill 2
    trajectory_2 = trajectory.clean(output.best_distance_infill_2, output.best_speed_infill_2,
                                    accel_2, plot_max_distance)
    # Trajektorie bei Aufwertung an Infill 3
    trajectory_3 = trajectory.clean(output.best_distance_infill_3, output.best_speed_infill_3,
                                    accel_3, plot_max_distance)
    # Trajektorie bei Aufwertung am Target
    trajectory_4 = trajectory.clean(output.distance_target, output.speed_target, accel_4,
                                    plot_max_distance)
    # Balisengruppen
    balises = np.array([0, output.infill_distance_1, input.track_infill_1])
    if input.track_balises == 3:
        balises = np.append(balises, output.infill_distance_2)
    balises = sorted(balises)
    balises_plot = totals.track_distance_origin_target - balises
    # Auflösung
    px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    plt.subplots(figsize=(1920*px, 1080*px))
    # Trajektorien
    try:
        plt.plot(trajectory_1[:, 0] / constants.SCALE-offset, trajectory_1[:, 1], color="green")
        plt.plot(trajectory_2[:, 0] / constants.SCALE-offset, trajectory_2[:, 1], color="orange")
    except:
        raise NotImplementedError("unable to plot")
    try:
        plt.plot(trajectory_3[:, 0] / constants.SCALE-offset, trajectory_3[:, 1], color="red")
    except:
        pass
    try:
        plt.plot(trajectory_4[:, 0] / constants.SCALE-offset, trajectory_4[:, 1], color="purple")
    except:
        raise NotImplementedError("unable to plot")
    # Indication Point mit Beschriftung
    indication_plot_x = totals.track_distance_origin_target-input.train_indication_point-offset
    plt.scatter(indication_plot_x, max_speed, s=80, marker="X", color="orange")
    plt.annotate("IP", (indication_plot_x, max_speed), textcoords="offset points", xytext=(0, 7),
                 ha="center")
    # Balisengruppen als Dreiecke
    plt.scatter(balises_plot-offset, [0]*len(balises_plot), s=100, marker=mpl.markers.CARETUPBASE,
                color="black")
    # senkrechte Linien über Balisengruppen
    plt.vlines(balises_plot-offset, 0, max_speed+2.5, color="grey", linestyles="dashed")
    # Bezeichnung der Balisengruppen
    for x in range(len(balises_plot)):
        name = "EoA" if x == 0 else f"IF_{x}"
        plt.annotate(name, (balises_plot[x]-offset, max_speed+2.5), textcoords="offset points",
                     xytext=(0, 7), ha="center")

    # senkrechte Linie wenn letzte Trajektorie die Ausgangsgeschwindigkeit wieder erreicht hat
    plt.vlines(np.max(output.distance_infill_1)-offset, 0, plot_max_speed, color="grey",
               linestyles="dotted")
    # Achsen, Ticks und Achsenabel
    plt.axis([-offset, plot_max_distance-offset, 0, plot_max_speed])
    plt.xticks(np.arange(0-offset, plot_max_distance-offset, step=250))
    plt.yticks(np.arange(0, plot_max_speed, step=10))
    if locale == LOC_DE:
        plt.xlabel("Strecke [$m$]")
        plt.ylabel("Geschwindigkeit [$km/h$]")
    elif locale == LOC_EN:
        plt.xlabel("distance [$m$]")
        plt.ylabel("speed [$km/h$]")

    # Text für Ausgabe der Stufenfunktionen von Anfahren und Bremsen
    label_accel = ""
    label_decel = ""
    for x in range(len(input.train_acceleration[1])):
        if x < len(input.train_acceleration[1])-1:
            label_accel += f"{input.train_acceleration[0,x]*3.6:.1f} $km/h$ "\
                f"< {input.train_acceleration[1,x+1]:.2f} $m/s^2$ > "
        else:
            label_accel += f"{input.train_acceleration[0,x]*3.6:.1f} $km/h$"
    for x in range(len(input.train_deceleration[1])-1, -1, -1):
        if x > 0:
            label_decel += f"{input.train_deceleration[0,x]*3.6:.1f} $km/h$ "\
                f"< {input.train_deceleration[1,x]:.2f} $m/s^2$ > """
        else:
            label_decel += f"{input.train_deceleration[0,x]*3.6:.1f} $km/h$"
    if len(label_accel.replace("$", "")) > 150:
        if locale == LOC_DE:
            label_accel = "zu viele Werte"
        elif locale == LOC_EN:
            label_accel = "too many values"
    if len(label_decel.replace("$", "")) > 150:
        if locale == LOC_DE:
            label_decel = "zu viele Werte"
        elif locale == LOC_EN:
            label_decel = "too many values"
    # Label festgelegte Balisengruppen
    label_fixed = f"{input.track_infill_3}, {input.track_infill_2}, {input.track_infill_1}"
    label_fixed = label_fixed.replace(" 0, ", " ")
    # Label Position der Balisengruppen
    match input.track_balises:
        case 2:
            label_positions = f"{output.infill_distance_1} & {input.track_infill_1}"
            pass
        case 3:
            label_positions = (f"{output.infill_distance_2}, {output.infill_distance_1} & "
                               f"{input.track_infill_1}")
    # Label Zeitverlust nach Balisenanzahl
    label_timeloss_1 = ""
    label_timeloss_2 = ""
    if input.track_balises == 3:
        if locale == LOC_DE:
            label_timeloss_1 = (f"\nZeitverlust bei Aufwertung {output.infill_distance_2} $m$ vor "
                                f"EoA: {output.best_delta_infill_3:.2f} $s$\n")
            label_timeloss_2 = f"Gewicht: {factors[1]:.3f}\n"
        elif locale == LOC_EN:
            label_timeloss_1 = (f"\nadditional run time with infill at {output.infill_distance_2} "
                                f"$m$ in rear of EoA: {output.best_delta_infill_3:.2f} $s$\n")
            label_timeloss_2 = f"weight: {factors[1]:.3f}\n"
    # Text mit Berechungsparametern
    if locale == LOC_DE:
        label = f"""
        PARAMETER STRECKE\n
        Geschwindigkeit: {input.track_line_speed:.0f} $km/h$\n
        Release Speed: {input.track_release_speed:.0f} $km/h$\n
        Neigung: {input.track_gradient:.1f} """ + u"\u2030" + f"""\n
        Anzahl Infill-Balisengruppen: {input.track_balises}\n
        Festlegte Infill-Balisengruppe(n) {label_fixed} $m$ vor EoA\n
        Mindestabstand Balisengruppen: {input.track_balise_group_distance} $m$\n
        \n
        PARAMETER ZUG\n
        Geschwindigkeit: {input.train_speed:.0f} $km/h$\n
        Rotierende Massen: {input.train_rotating_mass:.1f} %\n
        Indication Point: {input.train_indication_point} $m$\n
        Minimale Beharrungsfahrt: {input.train_min_cruise_time:.1f} $s$\n
        Verarbeitungszeit: {input.train_processing_time:.1f} $s$\n
        Bremsbeschleunigungen:\n
        {label_decel}\n
        Anfahrbeschleunigungen:\n
        {label_accel}\n
        \n
        ERGEBNISSE\n
        Infill-BG {label_positions} $m$ vor EoA\n
        Zeitverlust bei Aufwertung 0 $m$ vor EoA: {output.delta_target:.2f} $s$\n
            Gewicht: {factors[0]:.3f}
        {label_timeloss_1}
        {label_timeloss_2}
        Zeitverlust bei Aufwertung {output.infill_distance_1} $m$ vor EoA: {
            output.best_delta_infill_2:.2f} $s$\n
            Gewicht: {factors[-1]:.3f}\n
        Gewichtungsmethode: {input.tech_weighting.name.title()}\n
        gewichteter Zeitverlust: {output.min_loss_time:.2f} $s$\n
        """
    elif locale == LOC_EN:
        label = f"""
        PARAMETERS INFRASTRUCTURE\n
        line speed: {input.track_line_speed:.0f} $km/h$\n
        release speed: {input.track_release_speed:.0f} $km/h$\n
        gradient: {input.track_gradient:.1f} """ + u"\u2030" + f"""\n
        number of infill balise groups: {input.track_balises}\n
        fixed infill balise group(s) {label_fixed} $m$ vor EoA\n
        minimum distance between balise groups: {input.track_balise_group_distance} $m$\n
        \n
        PARAMETERS TRAIN\n
        maximum speed: {input.train_speed:.0f} $km/h$\n
        rotating masses: {input.train_rotating_mass:.1f} %\n
        indication point: {input.train_indication_point} $m$\n
        minimum cruising time: {input.train_min_cruise_time:.1f} $s$\n
        processing time: {input.train_processing_time:.1f} $s$\n
        deceleration:\n
        {label_decel}\n
        acceleration:\n
        {label_accel}\n
        \n
        RESULTS\n
        infill balise groups {label_positions} $m$ in rear of EoA\n
        additional run time with infill at 0 $m$ in rear of EoA: {output.delta_target:.2f} $s$\n
            weight: {factors[0]:.3f}
        {label_timeloss_1}
        {label_timeloss_2}
        additional run time with infill at {output.infill_distance_1} $m$ in rear of EoA: {
            output.best_delta_infill_2:.2f} $s$\n
            weight: {factors[-1]:.3f}\n
        weighting method: {input.tech_weighting.name.title()}\n
        weighted additional run time: {output.min_loss_time:.2f} $s$\n
        """
    label = label.replace("nan, ", "")
    plt.annotate(label, (plot_max_distance-35-offset, 0), ha="right", linespacing=0.6)
    # speichern
    path = "output"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(
        f"./{path}/{input.timestr}_trajectory.{FILETYPE}", bbox_inches="tight", pad_inches=0.5)

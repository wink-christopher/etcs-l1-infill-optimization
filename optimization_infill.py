"""
Version 0.11
Build on Python 3.11.9 with (see requirements.txt)
Contact: wink@via.rwth-aachen.de
Change History:
- 0.11, 2024-04-08 cw: Einheitliche Dateinamenpräfixe & PEP 8 Konformität
- 0.10, 2024-04-04 cw: Bugfix Berücksichtigung Gradiente
- 0.09, 2024-04-02 cw: Lokalisierung
- 0.08, 2024-03-25 cw: Kommentare & Codeoptimierung
- 0.07, 2024-02-26 cw: Codeoptimierung
- 0.06, 2023-07-27 cw: Änderung cwd, try für Parameter-Datei laden und parsen
- 0.05, 2023-07-26 cw: Fixes Unterscheidung 2/3 IF-BG
- 0.04, 2023-07-21 cw: diveres Fixes & Berechnung mit drei vorgegebenen IF-BG möglich
- 0.03, 2023-07-13 cw: fertige Dokumentation
- 0.02, 2023-07-12 cw: Erweiterte Dokumentation und Optimierungen
- 0.01, 2023-07-11 cw: Erster Stand mit begonnener Dokumentation
"""

import enum
import json
import logging
import numpy as np
import os
import pandas as pd
import time

import calc_movements as calc
import checks
import constants
import plots


logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} -> [%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    os.chdir("./infill_optimization")
except:
    pass


class Weighting(enum.Enum):
    """"
    Beinhaltet die möglichen Gewichtungsmethoden zur Berechnung des gewichteten Fahrzeitverlusts.
    """

    TIME = 1  # Abschnitte nach ihrer Fahrzeit [s] gewichten
    DISTANCE = 2  # Abschnitte nach ihrer Distanz [m] gewichten
    EQUAL = 3  # Abschnitt gleich gewichten


class Input:
    """"
    Liest die Parameter-JSON ein und hält die Input-Parameter in Variablen vor.
    """

    # Datei parameter.json laden
    try:
        file = open("./parameters.json")
        input_data = json.load(file)
        file.close()
    except:
        logger.error("Datei 'parameters.json' wurde nicht gefunden.")
        input("Enter zum Beenden...")
        exit()

    # Parameter im Abschnitt 'track' laden
    try:
        input_track = input_data["track"]
        track_line_speed = input_track["line_speed"]
        track_release_speed = input_track["release_speed"]
        track_gradient = input_track["gradient"]
        track_balises = input_track["balises"]
        track_balise_group_distance = input_track["balise_group_distance"]
        track_balise_positions = input_track["balise_positions"]
        track_balise_positions.sort(reverse=True)
        track_infill_1 = track_balise_positions[0]
        track_infill_2 = track_balise_positions[1]
        track_infill_3 = track_balise_positions[2] if track_balises > 2 else np.nan
    except:
        logger.error("Parameter 'track' konnten nicht alle geladen werden.")
        input("Enter zum Beenden...")
        exit()

    # Parameter im Abschnitt 'train' laden
    try:
        input_train = input_data["train"]
        train_speed = input_train["speed"]
        train_deceleration = np.array([input_train["deceleration"]["steps"],
                                       input_train["deceleration"]["values"]])
        train_acceleration = np.array([input_train["acceleration"]["steps"],
                                       input_train["acceleration"]["values"]])
        train_rotating_mass = input_train["rotating_mass"]
        train_indication_point = input_train["indication_point"]
        train_min_cruise_time = input_train["min_cruise_time"]
        train_processing_time = input_train["processing_time"]
    except:
        logger.error("Parameter 'train' konnten nicht alle geladen werden.")
        input("Parameter 'train' konnten nicht alle geladen werden.")
        exit()

    # Parameter im Abschnitt 'tech' laden
    try:
        input_tech = input_data["tech"]
        tech_steps = input_tech["steps"]
        tech_weighting = Weighting[input_tech["weighting"]]
        tech_plot_2d = input_tech["plot_trajectories"]
        tech_plot_3d = input_tech["plot_3d"]
        tech_rotate_plot = input_tech["rotate_plot"] if tech_plot_3d else False
        tech_locale = input_tech["locale"]
    except:
        logger.error("Parameter 'tech' konnten nicht alle geladen werden.")
        input("Parameter 'tech' konnten nicht alle geladen werden.")
        exit()

    timestr = time.strftime('%Y%m%d-%H%M%S')


class Totals:
    """"
    Hält die Variablen verrechneter Eingangsgrößen vor.
    """

    # Kopie des input erstellen
    train_deceleration = Input.train_deceleration
    train_acceleration = Input.train_acceleration
    # Geschwindigkeiten in m/s konvertieren
    train_deceleration[0] = train_deceleration[0]*constants.CONVERT_KPH_MPS
    train_acceleration[0] = train_acceleration[0]*constants.CONVERT_KPH_MPS
    train_speed = np.minimum(Input.track_line_speed, Input.train_speed) * constants.CONVERT_KPH_MPS
    track_release_speed = Input.track_release_speed * constants.CONVERT_KPH_MPS
    # Beschleunigungen mit rotierenden Massen und Gradiente korrigieren
    train_deceleration[1, 1:] = (Input.train_deceleration[1, 1:]
                                 - constants.G/(1+Input.train_rotating_mass/100)
                                 * Input.track_gradient/1000)
    train_acceleration[1, 1:] = (Input.train_acceleration[1, 1:]
                                 - constants.G/(1+Input.train_rotating_mass/100)
                                 * Input.track_gradient/1000)
    # Rundung des Betrachtungsraumes
    track_distance_origin_target = np.ceil((np.maximum(Input.track_infill_1,
                                                       Input.train_indication_point)+1)/250) * 250


class Output:
    """"
    Hält die Ausgabewerte in Variablen vor.
    """

    results = np.empty((Input.track_infill_1, Input.track_infill_1))
    results[:] = np.nan
    distance_infill_1 = []
    best_distance_infill_2 = []
    best_distance_infill_3 = []
    distance_target = []
    speed_infill_1 = []
    best_speed_infill_2 = []
    best_speed_infill_3 = []
    speed_target = []
    delta_target = 0
    best_delta_infill_2 = 0
    best_delta_infill_3 = 0
    min_loss_time = float("inf")
    infill_distance_1 = float("inf")
    infill_distance_2 = float("inf")


# Speicherort der Fahrzeiten vor/zwischen/nach den Balisengruppen
match Input.track_balises:
    case 2: running_time_intervals = [0, 0, 0, 0]
    case 3: running_time_intervals = [0, 0, 0, 0, 0]


def infill_at_target(distance_limit: float) -> tuple[float, float, np.ndarray, np.ndarray,
                                                     np.ndarray]:
    """
    Berechnet die Trajektorie bei Aufwertung an der letzten Balisengruppe am End of Authority.

    Args:
        distanceLimit: maximal zur Verfügung stehende Distanz für den Geschwindigkeitswechsel in m

    Raises:
        ValueError: Distanzlimit negativ
        ValueError: Zielgeschwindigkeit in Distanzlimit nicht erreichbar

    Returns:
        s_total: Distanz bis zum Wiedererreichen der zulässigen Geschwindigkeit in m
        t_total: Zeit bis zum Wiederreichen der zulässigen Geschwindigkeit in s
        distance_info: markante Punkte - Distanzen in m
        speed_info: markante Punkte - Geschwindigkeiten in m/s
        accel_info: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if distance_limit < 0:
        raise ValueError(f"Wert für 'distance_limit' negativ ({distance_limit} m)")
    # Initialisierung
    distance_info = [0]
    speed_info = [Totals.train_speed]
    accel_info = []
    # Beharrungsfahrt bis Indication Point
    s_approach, t_approach = calc.cruise(Totals.track_distance_origin_target
                                         - Input.train_indication_point, Totals.train_speed, 0)
    # Bremsen bis auf Release Speed zwischen Indication Point und EoA
    s_decel, t_decel, s_decel_steps, v_decel_steps, a_decel_steps = calc.speed_change_open(
        Totals.train_speed, Totals.track_release_speed, Totals.train_deceleration)
    # Prüfung ob Zielgeschwindigkeit überhaupt erreichbar ist
    if s_decel > distance_limit:
        raise ValueError("Zielgeschwindigkeit nicht erreichbar")
    # Beharrungsfahrt mit Release Speed bis Balisengruppe am EoA
    s_release, t_release = calc.cruise(Input.train_indication_point-s_decel,
                                       Totals.track_release_speed, Input.train_min_cruise_time)
    # Beharrungsfahrt nach Balisengruppe am EoA
    s_process, t_process = calc.processing(Totals.track_release_speed, Input.train_processing_time)
    # Beschleunigung auf Ausgangsgeschwindigkeit
    s_accel, t_accel, s_accel_steps, v_accel_steps, a_accel_steps = calc.speed_change_open(
        Totals.track_release_speed, Totals.train_speed, Totals.train_acceleration)
    # Beharrungsfahrt mit Release Speed deckt bereits Processing mit ab
    if s_decel+s_release-Input.train_indication_point >= s_process:
        s_process = 0
        t_process = 0

    # Summen von Strecke und Zeit
    s_total = s_approach + s_decel + s_release + s_process + s_accel
    t_total = t_approach + t_decel + t_release + t_process + t_accel
    # markante Punkte speichern
    distance_info.append(distance_info[-1] + s_approach)
    speed_info.append(Totals.train_speed)
    accel_info.append(0)
    distance_info.extend(distance_info[-1] + np.array(s_decel_steps[1:]))
    speed_info.extend(v_decel_steps[1:])
    accel_info.extend(a_decel_steps)
    distance_info.append(distance_info[-1] + s_release)
    speed_info.append(Totals.track_release_speed)
    accel_info.append(0)
    distance_info.append(distance_info[-1] + s_process)
    speed_info.append(Totals.track_release_speed)
    accel_info.append(0)
    distance_info.extend(distance_info[-1] + np.array(s_accel_steps[1:]))
    speed_info.extend(v_accel_steps[1:])
    accel_info.extend(a_accel_steps)
    # Fahrzeit der Trajektorie speichern
    running_time_intervals[-1] = t_approach + t_decel + t_release
    # Logging
    logger.debug(f"s_approach = {s_approach:.2f} m")
    logger.debug(f"s_decel = {s_decel:.2f} m")
    logger.debug(f"s_release = {s_release:.2f} m")
    logger.debug(f"s_process = {s_process:.2f} m")
    logger.debug(f"s_accel = {s_accel:.2f} m")
    logger.debug(f"s_total_target = {s_total:.2f} m")
    logger.debug(f"t_approach = {t_approach:.2f} s")
    logger.debug(f"t_decel = {t_decel:.2f} s")
    logger.debug(f"t_release = {t_release:.2f} s")
    logger.debug(f"t_process = {t_process:.2f} s")
    logger.debug(f"t_accel = {t_accel:.2f} s")
    logger.debug(f"t_total_target = {t_total:.2f} s")
    # Einheitenkonertierung und Rundung
    speed_info = np.round(np.array(speed_info)*constants.CONVERT_MPS_KPH, 2)
    distance_info = np.round(distance_info, 2)

    return s_total, t_total, distance_info, speed_info, accel_info


def infill_in_rear_of_IP(s_target: float, speed: float) -> tuple[float, float, np.ndarray,
                                                                 np.ndarray, np.ndarray]:
    """
    Berechnet die Trajektorie eines Zuges, der ungehindert fährt, weil die Aufwertung vor dem
    Indication Point erfolgt.

    Args:
        s_total_target: Distanz bis zum Wiedererreichen der zulässigen Geschwindigkeit in m
        speed: gefahrene Geschwindigkeit in m/s

    Raises:
        ValueError: Distanz negativ
        ValueError: Geschwindigkeit negativ

    Returns:
        s_total_infill_1: gesamte gefahrene Strecke in m
        t_total_infill_1: gesamte Fahrzeit in s
        distance_info: markante Punkte - Distanzen in m
        speed_info: markante Punkte - Geschwindigkeiten in m/s
        accel_info: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if s_target < 0:
        raise ValueError(f"Wert für 's_target' negativ ({s_target} m)")
    if speed < 0:
        raise ValueError(f"Wert für 'speed' negativ ({speed} m/s)")
    # Initialisierung
    distance_info = [0]
    speed_info = [Totals.train_speed]
    accel_info = []
    # Summen von Strecke und Zeit
    s_total, t_total = calc.cruise(s_target, speed, 0)
    # markante Punkte speichern
    distance_info.append(distance_info[-1]+s_total)
    speed_info.append(Totals.train_speed)
    accel_info.append(0)
    # Fahrzeit der Trajektorie speichern
    running_time_intervals[1] = calc.cruise(
        Totals.track_distance_origin_target-Input.track_infill_1, speed, 0)[1]
    # Logging
    logger.debug(f"t_total_infill_1 = {t_total:.2f} s")
    logger.debug(f"s_total_infill_1 = {s_total:.2f} m")
    # Einheitenkonertierung und Rundung
    speed_info = np.round(np.array(speed_info)*constants.CONVERT_MPS_KPH, 2)
    distance_info = np.round(distance_info, 2)

    return s_total, t_total, distance_info, speed_info, accel_info


def infill_in_advance_of_IP(distance_1: int, s_target: float, counter: int
                            ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet die Trajektorie eines Zuges, der an seinem Indication Point einen Bremsvorgang
    einleitet und an einem beliebigen Punkt danach eine Aufwertung der MA erhält. Danach
    beschleunigt er wieder auf die gewünschte Geschwindigkeit.

    Args:
        distance_1: Position der Infill-Balisengruppe vor dem EoA
        s_total_target: Distanz bis zum Wiedererreichen der zulässigen Geschwindigkeit in m
        counter: Abschnittsnummer vor dem Punkt der Aufwertung

    Raises:
        ValueError: Infill-Distanz negativ
        ValueError: gesamte Fahrstrecke negativ

    Returns:
        s_total: gesamte gefahrene Strecke in m
        t_total: gesamte Fahrzeit in s
        distance_info: markante Punkte - Distanzen in m
        speed_info: markante Punkte - Geschwindigkeiten in m/s
        accel_info: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if distance_1 < 0:
        raise ValueError(f"Wert für 'distance_1' negativ ({distance_1} m)")
    if s_target < 0:
        raise ValueError(f"Wert für 's_target' negativ ({s_target} m)")
    # Initialisierung
    distance_info = [0]
    speed_info = [Totals.train_speed]
    accel_info = []
    # Beharrungsfahrt bis Indication Point
    s_approach, t_approach = calc.cruise(
        Totals.track_distance_origin_target-Input.train_indication_point, Totals.train_speed, 0)
    # Bremsen von Indication Point bis Infill-Balisengruppe
    s_decel, t_decel, infill_speed, s_decel_steps, v_decel_steps, \
        a_decel_steps = calc.speed_change_limit(Totals.train_speed, Totals.track_release_speed,
                                                Totals.train_deceleration,
                                                Input.train_indication_point-distance_1)
    # Bremsen von Infill-Balisengruppe bis Ende Verarbeitungszeit
    s_process, t_process, process_speed, cruise_time, s_process_steps, v_process_steps, \
        a_process_steps = calc.speed_change_fixed_time(infill_speed, Totals.track_release_speed,
                                                       Totals.train_deceleration,
                                                       Input.train_processing_time,
                                                       Input.train_processing_time)
    # Beharrungsfahrt zwischen Bremsen und Beschleunigen
    s_release, t_release = calc.cruise(
        np.maximum(Input.train_indication_point-distance_1-s_decel-s_process, 0), process_speed,
        np.maximum(Input.train_min_cruise_time-cruise_time, 0))
    # Beschleunigen nach Aufwertung bis Ausgangsgeschwindigkeit
    s_accel, t_accel, s_accel_steps, v_accel_steps, a_accel_steps = calc.speed_change_open(
        process_speed, Totals.train_speed, Totals.train_acceleration)
    # Beharrungsfahrt bis Ende Betrachtungsraum
    s_cruise, t_cruise = calc.cruise(s_target-s_approach-s_decel-s_process-s_release-s_accel,
                                     Totals.train_speed, 0)
    # Summen von Strecke und Zeit
    s_total = s_approach + s_decel + s_process + s_release + s_accel + s_cruise
    t_total = t_approach + t_decel + t_process + t_release + t_accel + t_cruise
    # Fahrzeit der Trajektorie speichern
    running_time_intervals[counter+1] = t_approach + t_decel
    # markante Punkte speichern
    distance_info.append(distance_info[-1] + s_approach)
    speed_info.append(Totals.train_speed)
    accel_info.append(0)
    distance_info.extend(distance_info[-1] + np.array(s_decel_steps[1:]))
    speed_info.extend(v_decel_steps[1:])
    accel_info.extend(a_decel_steps)
    distance_info.extend(distance_info[-1] + np.array(s_process_steps[1:]))
    speed_info.extend(v_process_steps[1:])
    accel_info.extend(a_process_steps)
    distance_info.append(distance_info[-1] + s_release)
    speed_info.append(process_speed)
    accel_info.append(0)
    distance_info.extend(distance_info[-1] + np.array(s_accel_steps[1:]))
    speed_info.extend(v_accel_steps[1:])
    accel_info.extend(a_accel_steps)
    distance_info.append(distance_info[-1] + s_cruise)
    speed_info.append(Totals.train_speed)
    accel_info.append(0)
    # Logging
    logger.debug(f"Speed at Infill: {infill_speed*constants.CONVERT_MPS_KPH:.2f} km/h")
    logger.debug(f"Speed after Processing: {process_speed*constants.CONVERT_MPS_KPH:.2f} km/h")
    logger.debug(f"s_approach = {s_approach:.2f} m")
    logger.debug(f"s_decel = {s_decel:.2f} m")
    logger.debug(f"s_process = {s_process:.2f} m")
    logger.debug(f"s_release = {s_release:.2f} m")
    logger.debug(f"s_accel = {s_accel:.2f} m")
    logger.debug(f"s_cruise2 = {s_cruise:.2f} m")
    logger.debug(f"s_total_target = {s_target:.2f} m")
    logger.debug(f"t_approach = {t_approach:.2f} s")
    logger.debug(f"t_decel = {t_decel:.2f} s")
    logger.debug(f"t_process = {t_process:.2f} s")
    logger.debug(f"t_release = {t_release:.2f} s")
    logger.debug(f"t_accel = {t_accel:.2f} s")
    logger.debug(f"t_cruise = {t_cruise:.2f} s")
    logger.debug(f"t_total_infill_2 = {t_total:.2f} s")
    # Einheitenkonertierung und Rundung
    speed_info = np.round(np.array(speed_info)*constants.CONVERT_MPS_KPH, 2)
    distance_info = np.round(distance_info, 2)

    return s_total, t_total, distance_info, speed_info, accel_info


def optimize(balises: int, steps: int, fixed_1: int, fixed_2: int, envelope: int) -> tuple[int,
                                                                                           int]:
    """
    Führt die Optimierung der Balisenstandorte für vorgegebene Grenzen und Schrittweiten durch.
    Für jede berechnete Kombination wird der gewichtete Fahrzeitverlust bestimmt und immer die
    bisher beste Kombination (also mit dem geringsten gewichteten Fahrzeitverlust) gespeichert.
    Findet eine Berechnung mit Schrittweite 1 m statt, so wird im Anschluss das Ergebnis ausgegeben
    sowie das Plotten angestoßen

    Args:
        balises: Gesamtzahl der Infill-Balisengruppen
        steps: Schrittweite der Balisenpositionierung in m
        fixed_1: Vorgabe einer Infill-Balisengruppe in m vor dem EoA (0 = keine Vorgabe)
        fixed_2: Vorgabe einer Infill-Balisengruppe in m vor dem EoA (0 = keine Vorgabe)
        envelope: Suchugebung um Mittelpunkt aus fixed_? in m

    Raises:
        ValueError: Fahrzeitverlust negativ

    Returns:
        Output.infill_distance_1: Optimale Balisenposition Infill 1 vor dem EoA in m
        Output.infill_distance_2: Optimale Balisenposition Infill 2 vor dem EoA in m
    """

    # Initialisierung
    min_loss_prev = float("inf")
    best_accel_infill_3 = []
    best_factors = []
    # Grenzen setzen
    if Input.track_infill_2 > 0:  # zwei Infillbalisengruppen vorgegeben
        start_1 = Input.track_infill_2
        limit_1 = start_1 + 1
    else:  # nur eine Infillbalisengruppe vorgegeben
        start_1 = 1 + Input.track_balise_group_distance
        limit_1 = np.minimum(Input.train_indication_point, Input.track_infill_1
                             ) - Input.track_balise_group_distance
    # Grenzen setzen
    if fixed_1 > 0:
        if Input.track_infill_2 > 0:  # Position eine weiteren Balisengruppe ist vorgegeben
            start_1 = fixed_1
            limit_1 = fixed_1 + 1
        else:  # keine weitere Position ist vorgegeben
            start_1 = fixed_1 - envelope
            limit_1 = fixed_1 + envelope + 1

    # Trajektorie bei Infill an Balisengruppe am EoA
    s_total_target, t_total_target, Output.distance_target, Output.speed_target, \
        accel_target = infill_at_target(Input.track_infill_1)
    # Relation Indication Point zu erste Infillbalisengruppe
    if Input.train_indication_point > Input.track_infill_1:  # Regelfall
        t_total_infill_1, Output.distance_infill_1, Output.speed_infill_1, \
            accel_infill_1 = infill_in_advance_of_IP(Input.track_infill_1, s_total_target, 0)[1:]
    else:  # Indication Point noch vor erster Infillbalisengruppe
        t_total_infill_1, Output.distance_infill_1, Output.speed_infill_1, \
            accel_infill_1 = infill_in_rear_of_IP(s_total_target, Totals.train_speed)[1:]

    # Fahrzeitverlängerung bei Infill an Balisengruppe am EoA
    Output.delta_target = t_total_target - t_total_infill_1
    # Logging
    logger.debug(f"Delta Target: {Output.delta_target:.2f} s")
    # Schleife über Position der ersten freien Infillbalisengruppe
    for distance_1 in range(start_1, limit_1, steps):
        # Initialisierung
        min_loss_iter = float("inf")
        start_2 = 1
        # Grenzen setzen
        limit_2 = 1 + 1 if balises == 2 else distance_1 - Input.track_balise_group_distance + 1
        if fixed_2 > 0:
            start_2 = fixed_2 - envelope
            limit_2 = fixed_2 + envelope + 1

        # Schleife über Position der zweiten freien Infillbalisengruppe
        for distance_2 in range(start_2, limit_2, steps):
            # Trajektorie bei Aufwertung an zweiter freien Infillbalisengruppe berechnen
            t_total_infill_2, distance_infill_2, speed_infill_2, \
                accel_infill_2 = infill_in_advance_of_IP(distance_1, s_total_target, 1)[1:]
            delta_infill_2 = t_total_infill_2 - t_total_infill_1
            # Logging
            logger.debug(f"Delta Infill 1: {delta_infill_2:.2f} s")

            match balises:
                case 2:
                    # Gewichtungsfaktoren berechnen
                    factors = [0, 0]  # Target -> IF -> IF
                    match Input.tech_weighting:
                        case Weighting.TIME:
                            factors = [running_time_intervals[3] - running_time_intervals[2],
                                       running_time_intervals[2] - running_time_intervals[1]]
                        case Weighting.DISTANCE:
                            factors = [distance_1, Input.track_infill_1 - distance_1]
                        case Weighting.EQUAL:
                            factors = [1, 1]

                    # mittleren Fahrzeitverlust berechnen
                    mean_time_loss = (factors[0]*Output.delta_target + factors[1]*delta_infill_2
                                      ) / sum(factors)
                case 3:
                    # Trajektorie bei Aufwertung an zweiter freien Infillbalisengruppe berechnen
                    t_total_infill_3, distance_infill_3, speed_infill_3, \
                        accel_infill_3 = infill_in_advance_of_IP(distance_2, s_total_target, 2)[1:]
                    delta_infill_3 = t_total_infill_3 - t_total_infill_1
                    logger.debug(f"Delta Infill 2: {delta_infill_3:.2f} s")
                    # Gewichtungsfaktoren berechnen
                    factors = [0, 0, 0]  # Target -> IF -> IF
                    match Input.tech_weighting:
                        case Weighting.TIME:
                            factors = [running_time_intervals[4] - running_time_intervals[3],
                                       running_time_intervals[3] - running_time_intervals[2],
                                       running_time_intervals[2] - running_time_intervals[1],
                                       ]
                        case Weighting.DISTANCE:
                            factors = [distance_2,
                                       distance_1 - distance_2,
                                       Input.track_infill_1 - distance_1]
                        case Weighting.EQUAL:
                            factors = [1, 1, 1]

                    # mittleren Fahrzeitverlust berechnen
                    mean_time_loss = (factors[0]*Output.delta_target + factors[1]*delta_infill_3
                                      + factors[2]*delta_infill_2) / sum(factors)

            # Logging
            match balises:
                case 2:
                    logger.debug(f"Distance: {distance_1} m "
                                 f"-> Weighted Additional Runtime: {mean_time_loss:.2f} s")
                case 3:
                    logger.debug(f"Distances: {distance_2} m & {distance_1} m "
                                 f"-> Weighted Additional Runtime: {mean_time_loss:.2f} s")

            # Ergebnisse speichern wenn Verbesserung erreicht wird
            if mean_time_loss <= Output.min_loss_time:
                Output.min_loss_time = mean_time_loss
                Output.infill_distance_1 = distance_1
                Output.infill_distance_2 = distance_2
                Output.best_distance_infill_2 = distance_infill_2
                Output.best_speed_infill_2 = speed_infill_2
                Output.best_delta_infill_2 = delta_infill_2
                best_accel_infill_2 = accel_infill_2
                best_factors = factors

                if balises == 3:
                    Output.best_distance_infill_3 = distance_infill_3
                    Output.best_speed_infill_3 = speed_infill_3
                    Output.best_delta_infill_3 = delta_infill_3
                    best_accel_infill_3 = accel_infill_3

            # aktueller mininimaler Fahrzeitverlust
            min_loss_iter = np.minimum(min_loss_iter, mean_time_loss)
            # Fehler bei negativem Fahrzeitverlust
            if mean_time_loss < 0:
                raise ValueError(f"Fahrzeitverlust negativ ({mean_time_loss} s)")
            # 2D-Array für Ergebnis
            if balises == 3:
                Output.results[distance_1-1, distance_2-1] = mean_time_loss

        # minimaler Zeitverlust der vorherigen Iteration
        min_loss_prev = np.minimum(min_loss_prev, min_loss_iter)

    # Output der Ergebnisse
    if steps == 1:
        list_infill = [Output.infill_distance_1, Input.track_infill_1]
        match balises:
            case 2:
                logger.info(f"Min. gewichteter Fahrzeitverlust: {Output.min_loss_time:.2f} s "
                            f"bei {Output.infill_distance_1} m & {Input.track_infill_1} m")
            case 3:
                list_infill.append(Output.infill_distance_2)
                logger.info(f"Min. gewichteter Fahrzeitverlust: {Output.min_loss_time:.2f} s "
                            f"bei {Output.infill_distance_2} m, {Output.infill_distance_1} m &"
                            f" {Input.track_infill_1} m")

        # json Output
        output_data = Input.input_data
        list_infill.sort(reverse=True)
        output_data["results"] = {
            "infill_positions": list_infill,
            "additional_runtime": round(Output.min_loss_time, 2)
        }
        path = "output/json"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"./{path}/{Input.timestr}_results.json", "w") as outfile:
            json.dump(output_data, outfile, indent=4)
        # Logging
        if Input.tech_plot_2d or Input.tech_plot_3d:
            logging.info("plotten...")
        loglevel = logging.getLogger().getEffectiveLevel()
        if logging.getLevelName(loglevel) != 'INFO':
            logging.getLogger().setLevel(logging.INFO)
        # plotten der 2D-Trajektorien
        if Input.tech_plot_2d:
            plots.plot_trajectory(accel_infill_1, best_accel_infill_2, best_accel_infill_3,
                                  accel_target, Input, Totals, Output, best_factors)
        # plotten des 3D-Fahrzeitverlusts bei drei Infillbalisengruppen
        if (Input.tech_plot_3d and balises == 3):
            plots.plot_3d_shape(pd.DataFrame(Output.results), Input, Output)
        logging.getLogger().setLevel(loglevel)

    return Output.infill_distance_1, Output.infill_distance_2


def main() -> None:
    """
    Einstiegspunkt der Optimierung. Es werden die notwendigen Checks durchgeführt, bevor in einem
    zweistufigen Verfahren die optimale Platzierung der Infill-Balisengruppen gefunden wird.

    Args:
        none

    Raises:
        none

    Returns:
        none
    """

    # Timer starten
    tic = time.perf_counter()
    # Hinweis auf nicht gute Platzierung der weitesten Infillbalisengruppe
    if Input.train_indication_point > Input.track_infill_1:
        logger.info("Indication Point liegt vor erstem Infillpunkt")
    # Informationen
    logger.info(f"Geschwindigkeit: {Totals.train_speed*constants.CONVERT_MPS_KPH:.2f} km/h")
    logger.info(f"Gewichtungsmethode: {Weighting(Input.tech_weighting).name}")
    # Datenchecks durchführen
    checks.checks(Input, Totals)

    # Unterscheidung ob ein Lauf oder zwei Läufe notwendig
    if (((Input.track_balises == 3)
        and (Input.track_infill_1*Input.track_infill_2*Input.track_infill_3 > 0))   # 1 Lauf
            or ((Input.track_balises == 2) and (Input.track_infill_1*Input.track_infill_2))):
        logger.info("Durchlauf 1 von 1")
        distance_1, distance_2 = optimize(balises=Input.track_balises, steps=1,
                                          fixed_1=Input.track_infill_2,
                                          fixed_2=Input.track_infill_3, envelope=0)
    else:  # 2 Läufe notwendig
        logger.info("Durchlauf 1 von 2")
        distance_1, distance_2 = optimize(balises=Input.track_balises, steps=Input.tech_steps,
                                          fixed_1=0, fixed_2=0, envelope=0)
        logger.info("Durchlauf 2 von 2")
        distance_1, distance_2 = optimize(balises=Input.track_balises, steps=1, fixed_1=distance_1,
                                          fixed_2=distance_2, envelope=Input.tech_steps)

    # Timer stoppen
    toc = time.perf_counter()
    # Abschluss
    logger.info(f"Dauer: {toc - tic:0.2f} Sekunden")
    input("Enter zum Beenden...")


if __name__ == "__main__":
    main()

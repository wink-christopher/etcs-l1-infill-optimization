"""
Version 1.03
Build on Python 3.11.9 with (see requirements.txt)
Contact: wink@via.rwth-aachen.de
Change History:
- 1.03, 2024-04-08 cw: PEP 8 Konformität
- 1.02, 2024-04-05 cw: Verbesserte ValueError-Meldungen
- 1.01, 2024-03-25 cw: Kommentare & Codeoptimierung
- 1.00, 2023-07-12 cw: Initialer Stand mit Dokumentation und Versionierung
"""

import math
import numpy as np


def cruise(distance: float, speed: float, time_minimum: float) -> tuple[float, float]:
    """
    Berechnet eine Fahrt mit konstanter Geschwindigkeit über definierte Strecke und Zeit.

    Args:
        distance: maximale Distanz in m
        speed: Geschwindigkeit in m/s
        time_minimum: minimale Fahrzeit in s

    Raises:
        ValueError: Distanz negativ
        ValueError: Geschwindigkeit negativ
        ValueError: Zeit negativ

    Returns:
        distance_travelled: gefahrene Distanz in m
        time_elapsed: verstrichene Zeit in s
    """

    # Datencheck
    if distance < 0:
        raise ValueError(f"Wert für 'distance' negativ ({distance} m)")
    if speed < 0:
        raise ValueError(f"Wert für 'speed' negativ ({speed} m/s)")
    if time_minimum < 0:
        raise ValueError(f"Wert für 'time_minimum' negativ ({time_minimum} s)")
    # Berechnung
    time_elapsed = max(distance/speed, time_minimum)
    distance_travelled = time_elapsed * speed

    return distance_travelled, time_elapsed


def processing(speed: float, time: float) -> tuple[float, float]:
    """
    Berechnet eine Fahrt mit konstanter Geschwindigkeit über definierte Zeit.

    Args:
        speed: Geschwindigkeit in m/s
        time: Verarbeitungszeit in s

    Raises:
        ValueError: Geschwindigkeit negativ
        ValueError: Zeit negativ

    Returns:
        distance_travelled: gefahrene Distanz in m
        time_elapsed: verstrichene Zeit in s
    """

    # Datencheck
    if speed < 0:
        raise ValueError(f"Wert für 'speed' negativ ({speed} m/s)")
    if time < 0:
        raise ValueError(f"Wert für 'time' negativ ({time} s)")
    # Berechnung
    distance_travelled, time_elapsed = cruise(0, speed, time)

    return distance_travelled, time_elapsed


def speed_change_open(initial_speed: float, target_speed: float, acceleration: np.ndarray
                      ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet einen Geschwindigkeitswechsel zwischen zwei Geschwindigkeiten ohne Begrenzung.

    Args:
        initial_speed: Ausgangsgeschwindigkeit in m/s
        target_speed: Zielgeschwindigkeit in m/s
        acceleration: Stufenfunktion der Beschleunigung in m/s^2 über m/s

    Raises:
        ValueError: Ausgangsgeschwindigkeit negativ
        ValueError: Zielgeschwindigkeit negativ
        ValueError: Zielgeschwindigkeit nicht erreichbar

    Returns:
        distance_travelled: gefahrene Distanz in m
        time_elapsed: verstrichene Zeit in s
        distance_steps: markante Punkte - Distanzen in m
        speed_steps: markante Punkte - Geschwindigkeiten in m/s
        accel_steps: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if initial_speed < 0:
        raise ValueError(f"Wert für 'initial_speed'  negativ ({initial_speed} m/s)")
    if target_speed < 0:
        raise ValueError(f"Wert für 'target_speed' negativ ({target_speed} m/s)")
    if np.max(acceleration[0, :]) < target_speed:
        raise ValueError(f"Zielgeschwindigkeit ({target_speed} m/s) nicht erreichbar")

    # Index in Liste der Beschleunigungen bestimmen
    if target_speed >= initial_speed:
        index = np.argmax(acceleration[0] > initial_speed)
    else:
        index = np.argmin(acceleration[0] < initial_speed)

    # Initialisierung
    speed = initial_speed
    time_elapsed = 0
    distance_travelled = 0
    distance_steps = [0]
    speed_steps = [initial_speed]
    accel_steps = []
    # Iteration bis Zielgeschwindigkeit erreicht ist
    while speed != target_speed:
        # Geschwindigkeitsdifferenz bis Zielgeschwindigkeit
        if target_speed >= initial_speed:
            delta_speed = np.minimum(acceleration[0, index], target_speed) - speed
        else:
            delta_speed = np.maximum(acceleration[0, index-1], target_speed) - speed

        # Zeit und Distanz bis Geschwindigkeitswechsel durchgeführt ist
        delta_time = delta_speed / acceleration[1, index]
        delta_distance = 0.5*acceleration[1, index]*(delta_time**2) + speed*delta_time
        # Geschwindigkeit nach Iteration
        if target_speed >= initial_speed:
            speed = np.minimum(acceleration[0, index], target_speed)
        else:
            speed = np.maximum(acceleration[0, index-1], target_speed)

        # Iterationsergebnisse sichern
        time_elapsed += delta_time
        distance_travelled += delta_distance
        distance_steps.append(distance_steps[-1] + delta_distance)
        speed_steps.append(speed)
        accel_steps.append(acceleration[1, index])
        # Index in Liste der Beschleunigungswerte
        if target_speed >= initial_speed:
            index += 1
        else:
            index -= 1

    return distance_travelled, time_elapsed, distance_steps, speed_steps, accel_steps


def speed_change_limit(initial_speed: float, target_speed: float, acceleration: np.ndarray,
                       distance_limit: float) -> tuple[float, float, float, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """
    Berechnet einen Geschwindigkeitswechsel zwischen zwei Geschwindigkeiten mit Distanzlimit. Mit
    Erreichen des Limit wird keine weitere Geschwindigkeitsänderung mehr vorgenommen. Wird die
    Zielgeschwindigkeit vor Erreichen des Distanzlimit erreicht, wird dort das Segement beendet.

    Args:
        initial_speed: Ausgangsgeschwindigkeit in m/s
        target_speed: Zielgeschwindigkeit in m/s
        acceleration: Stufenfunktion der Beschleunigung in m/s^2 über m/s
        distance_limit: Distanzlimit in m

    Raises:
        ValueError: Ausgangsgeschwindigkeit negativ
        ValueError: Zielgeschwindigkeit negativ
        ValueError: Distanzlimit negativ

    Returns:
        distance_travelled: gefahrene Distanz in m
        time_elapsed: verstrichene Zeit in s
        exit_speed: Geschwindigkeit am Ende (zwischen Ausgangs- und Zielgeschwindigkeit) in m/s
        distance_steps: markante Punkte - Distanzen in m
        speed_steps: markante Punkte - Geschwindigkeiten in m/s
        accel_steps: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if initial_speed < 0:
        raise ValueError(f"Wert für 'initial_speed' negativ ({initial_speed} m/s)")
    if target_speed < 0:
        raise ValueError(f"Wert für 'target_speed' negativ ({target_speed} m/s)")
    if distance_limit < 0:
        raise ValueError(f"Wert für 'distance_limit' negativ ({distance_limit} m)")

    # Vergleichsrechnung Geschwindigkeitswechsel ohne Restriktion der Distanz
    distance_travelled, time_elapsed, distance_steps, speed_steps, \
        accel_steps = speed_change_open(initial_speed, target_speed, acceleration)

    if distance_travelled <= distance_limit:  # Vorgegebene Restriktion ist nicht relevant
        exit_speed = target_speed
    else:  # vorgegebene Restruktion ist relevant
        # markante Punkte des offenen Geschwindigkeitswechsels einkürzen
        trunc = np.searchsorted(distance_steps, distance_limit)
        distance_steps = distance_steps[0:trunc]
        speed_steps = speed_steps[0:trunc]
        accel_steps = accel_steps[0:trunc]
        distance_travelled = distance_limit
        # Reststück berechnen
        delta_distance = distance_travelled - distance_steps[-1]
        delta_time = -speed_steps[-1]/accel_steps[-1] - math.sqrt(
            (speed_steps[-1]/accel_steps[-1])**2+2*delta_distance/accel_steps[-1])
        exit_speed = speed_steps[-1] + accel_steps[-1]*delta_time
        # Ende des Geschwindigkeitswechsels speichern
        speed_steps.append(exit_speed)
        distance_steps.append(distance_travelled)
        # Fahrzeit berechnen
        time_elapsed = 0
        for i in range(len(speed_steps)-1):
            time_elapsed = time_elapsed + (speed_steps[i+1]-speed_steps[i])/accel_steps[i]

    return distance_travelled, time_elapsed, exit_speed, distance_steps, speed_steps, accel_steps


def speed_change_fixed_time(initial_speed: float, target_speed: float, acceleration: np.ndarray,
                            time_fixed: float, processing_time: float) -> tuple[
                                float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet einen Geschwindigkeitswechsel zwischen zwei Geschwindigkeit mit Zeitlimit. Mit
    Erreichen des Limit wird keine weitere Geschwindigkeitsänderung mehr vorgenommen.  Wird die
    Zielgeschwindigkeit vor Erreichen des Zeitlimit erreicht, wird eine Beharrungsfahrt
    angenommen.

    Args:
        initial_speed: Ausgangsgeschwindigkeit in m/s
        target_speed: Zielgeschwindigkeit in m/s
        acceleration: Stufenfunktion der Beschleunigung in m/s^2 über m/s
        time_fixed: Zeitvorgabe für die Dauer des Vorgangs in s
        processing_time: Verarbeitungszeit der OBU in s

    Raises:
        ValueError: Ausgangsgeschwindigkeit negativ
        ValueError: Zielgeschwindigkeit negativ
        ValueError: Zeitlimit negativ

    Returns:
        distance_travelled: gefahrene Distanz in m
        time_elapsed: verstrichene Zeit in s
        exit_speed/target_speed: Geschwindigkeit am Ende (zwischen Ausgangs- und
            Zielgeschwindigkeit) in m/s
        time_cruise: Beharrungsfahrzeit nach Beenndigung des Geschwindigkeitswechsels in s
        distance_steps: markante Punkte - Distanzen in m
        speed_steps: markante Punkte - Geschwindigkeiten in m/s
        accel_steps: markante Punkte - Beschleunigungen in m/s^2
    """

    # Datencheck
    if initial_speed < 0:
        raise ValueError(f"Wert für 'initial_speed' negativ ({initial_speed} m/s)")
    if target_speed < 0:
        raise ValueError(f"Wert für 'target_speed' negativ ({target_speed} m/s)")
    if time_fixed < 0:
        raise ValueError(f"Wert für 'time_fixed' negativ ({time_fixed} s)")

    # Initialisierung
    distance_steps = [0]
    speed_steps = [initial_speed]
    accel_steps = []
    time_elapsed = time_fixed

    # Dauer 0s
    if time_fixed == 0:
        # keine Änderungen zu berechnen
        distance_travelled = 0
        exit_speed = initial_speed
        time_cruise = 0

        return distance_travelled, time_elapsed, exit_speed, time_cruise, distance_steps, \
            speed_steps, accel_steps

    # Zielgeschwindigkeit bereits erreicht
    if initial_speed == target_speed:
        # Als Beharrungsfahrt behandeln
        distance_travelled, time_elapsed = processing(initial_speed, time_fixed)
        time_cruise = time_elapsed
        # markante Punkte speichern
        distance_steps.append(distance_steps[-1] + distance_travelled)
        speed_steps.append(target_speed)
        accel_steps.append(0)

        return distance_travelled, time_elapsed, target_speed, time_cruise, distance_steps, \
            speed_steps, accel_steps

    # vollständigen Geschwindigkeitswechsel berechnen
    distance_speed_change, time_speed_change, distance_steps_change, speed_steps_change, \
        accel_steps_change = speed_change_open(initial_speed, target_speed, acceleration)

    if time_speed_change <= time_fixed:  # vollständiger Geschwindigkeitswechsel möglich
        # Beharrungsfahrt nach Ende Geschwindigkeitswechsel
        exit_speed = target_speed
        time_processing_remaining = processing_time - time_speed_change
        distance_process, time_process = processing(target_speed, time_processing_remaining)
        # markante Punkte speichern
        distance_steps.extend(distance_steps[-1] + np.array(distance_steps_change))
        speed_steps.extend(speed_steps_change)
        accel_steps.extend(accel_steps_change)
        distance_steps.append(distance_steps[-1] + distance_process)
        speed_steps.append(target_speed)
        accel_steps.append(0)
        # Summe aus beiden Bestandteilen
        distance_travelled = distance_speed_change + distance_process
        time_elapsed = time_speed_change + time_process
        time_cruise = time_process
    else:  # vollständiger Geschwindigkeitswechsel nicht möglich
        # Initialisierung
        time_cruise = 0
        incremental_time = 0
        test_time = 0
        trunc = 0
        # Iteration bis Zeitlimit erreicht ist
        while test_time < time_fixed:
            incremental_time = test_time
            test_time = incremental_time + ((speed_steps_change[trunc+1]-speed_steps_change[trunc])
                                            / accel_steps_change[trunc])
            trunc += 1

        # markante Punkte berechnen
        distance_steps_change = distance_steps_change[0:trunc]
        speed_steps_change = speed_steps_change[0:trunc]
        accel_steps_change = accel_steps_change[0:trunc]
        delta_time = time_fixed - incremental_time
        delta_speed = speed_steps_change[-1] + accel_steps_change[-1]*delta_time
        delta_distance = (distance_steps_change[-1] + speed_steps_change[-1]*delta_time
                          + 0.5*accel_steps_change[-1]*(delta_time**2))
        # markante Punkte speichern
        distance_steps_change.append(delta_distance)
        speed_steps_change.append(delta_speed)
        distance_steps = distance_steps_change
        speed_steps = speed_steps_change
        accel_steps = accel_steps_change
        # Zustand nach Ablauf des Zeitlimits
        distance_travelled = distance_steps[-1]
        exit_speed = speed_steps[-1]

    return distance_travelled, time_elapsed, exit_speed, time_cruise, distance_steps, \
        speed_steps, accel_steps

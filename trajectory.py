"""
Version 1.02
Build on Python 3.11.9 with (see requirements.txt)
Contact: wink@via.rwth-aachen.de
Change History:
- 1.02, 2024-04-08 cw: PEP 8 Konformität
- 1.01, 2023-07-26 cw: Behandlung bei leerem Inputparameter
- 1.00, 2023-07-12 cw: Initialer Stand mit Dokumentation und Versionierung
"""

import numpy as np

import constants


def clean(points_distance: np.ndarray, points_speed: np.ndarray, points_accel: np.ndarray,
          plot_distance: float) -> np.ndarray:
    """
    Erzeugt eine Liste von Punkten für ein Geschwindigkeits-Weg-Diagramm.

    Args:
        points_distance: markante Punkte - Distanzen in m
        points_speed: markante Punkte - Geschwindigkeiten in m/s
        points_acceel: markante Punkte - Beschleunigungen in m/s^2
        plot_distance: gesamte zu plottende Strecke in m

    Raises:
        KeyError: Anzahl Elemente für Distanzen und Geschwindigkeiten nicht konsistent
        KeyError: Anazhl Elemente für Beschleunigungen und Geschwindigkeiten nicht konsistent

    Returns:
        data: Liste feingranularer Geschwindigkeiten in km/h über die Wegstrecke in cm
    """

    if (len(points_distance) * len(points_speed) * len(points_accel) * len(points_distance) == 0):
        return np.nan
    # Checks
    if len(points_distance) != len(points_speed):
        raise KeyError("ungleiche Anzahl Elemente für Distanz und Geschwindigkeit")
    if len(points_distance)-1 != len(points_accel):
        raise KeyError("unpassende Anzahl Elemente für Beschleunigung")
    # Initialisierung
    data = np.array(np.zeros((int(np.max(points_distance)*constants.SCALE) + 2, 2)))
    data[0:-1, 0] = np.arange(0, len(data)-1, 1)
    # Daten konvertieren und anpassen
    for change in range(0, len(points_speed)-1):
        if points_speed[change+1] == points_speed[change]:  # gleichbleibend
            data[int(points_distance[change]*constants.SCALE):int(points_distance[change+1] *
                                                                  constants.SCALE)+1,
                 1] = points_speed[change]
        else:  # bremsen oder beschleunigen
            data[int(points_distance[change]*constants.SCALE):int(points_distance[change+1] *
                                                                  constants.SCALE)+1,
                 1] = constants.CONVERT_MPS_KPH*np.sqrt(
                (points_speed[change]*constants.CONVERT_KPH_MPS)**2
                + 2*points_accel[change]*(data[int(
                    points_distance[change]*constants.SCALE):int(points_distance[change+1] *
                                                                 constants.SCALE)+1, 0]
                                                                 / constants.SCALE
                                                                 - points_distance[change]))
    # Endpunkt bestimmen
    data[-1, 0] = plot_distance*constants.SCALE
    data[-1, 1] = np.max(points_speed)

    return data

"""
Version 1.02
Build on Python 3.11.9 with (see requirements.txt)
Contact: wink@via.rwth-aachen.de
Change History:
- 1.02, 2024-04-08 cw: PEP 8 Konformität
- 1.01, 2024-04-05 cw: Zusätzliche Prüfung, ob zwischen IP & EOA auf 0 km/h gebremst werden kann
- 1.00, 2023-07-12 cw: Initialer Stand mit Dokumentation und Versionierung
"""

import numpy as np

import calc_movements as calc


def checks(input, totals) -> None:
    """
    Überprüfung der Input-Parameter auf Plausbilität

    Args:
        input: Klasse der Input-Parameter
        totals: Klasse der verrechneten Parameter

    Raises:
        ValueError: Anzahl Balisengruppen nicht 2 oder 3
        ValueError: Bremsbeschleunigung größer als 0
        ValueError: Anfahrbeschleunigung kleiner als 0
        ValueError: Mindestbeharrungsfahrzeit negativ
        ValueError: Verarbeitungszeit negativ
        ValueError: resultierende Bremsbeschleunigung größer als 0
        ValueError: resultierende Anfahrbeschleunigung kleiner als 0
        ValueError: Indication Point und Bremsbeschleunigung nicht kompatibel

    Returns:
        none
    """

    # direkte Prüfung der Inputdaten
    if input.track_balises not in [2, 3]:
        raise ValueError("Anzahl zusätzlicher Infill-Balisengruppen nicht unterstützt")
    if np.max(input.train_deceleration[1]) > 0:
        raise ValueError("Bremsbeschleunigung größer als 0")
    if np.min(input.train_acceleration[1]) < 0:
        raise ValueError("Anfahrbeschleunigung kleiner als 0")
    if input.train_min_cruise_time < 0:
        raise ValueError(f"Mindestbeharrungsfahrzeit negativ ({input.train_min_cruise_time} s)")
    if input.train_processing_time < 0:
        raise ValueError(f"Verarbeitungszeit negativ ({input.train_processing_time} s)")

    # Prüfung der verarbeiteten Inputdaten
    if np.max(totals.train_deceleration[1]) > 0:
        raise ValueError("Bremsvermögen zu gering oder Gefälle zu groß")
    if np.min(totals.train_acceleration[1]) < 0:
        raise ValueError("Anfahrvermögen zu gering oder Steigung zu groß")

    # Prüfung, ob ab IP Bremsung bis 0 km/h vor EOA möglich ist
    if calc.speed_change_open(totals.train_speed, 0,
                              totals.train_deceleration)[0] > input.train_indication_point:
        raise ValueError("Bremsung auf 0 km/h ab IP nicht möglich. Parameter überprüfen")

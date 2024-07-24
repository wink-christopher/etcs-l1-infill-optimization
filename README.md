# Determining the Optimal Positions of Infill Balise Groups for ERTMS/ETCS Level 1 Applications

## Overview
### Introduction
This Python program optimizes the positions of infill balise groups used for the [European Train Control System (ETCS)](https://en.wikipedia.org/wiki/European_Train_Control_System) when operating in Level 1 and mode "Full Supervision".
The optimization goal is to minimize the operational impact by determining the locations of infill balise groups which can transmit and update a Movement Authoritay (MA) to the train.
As the KPI we use the weighted additional runtime which is defined as the delay a train endures when the next section of track is not free and available and it has to initiate braking.

### Further Information
The corresponding research paper covering the algorithm and its input parameters in detail will be submitted to the journal "[Applied Sciences](https://www.mdpi.com/journal/applsci)" by MDPI.

## Usage
The programm is executed by runnig the "infill_optimization.py" file.
It will load the scenario parameters of the file "parameters.json" and write all output files to the "./output/" folder.

## Contributing
### Bugs
If you find a bug, please [open an issue](https://github.com/wink-christopher/etcs-l1-infill-optimization/issues).

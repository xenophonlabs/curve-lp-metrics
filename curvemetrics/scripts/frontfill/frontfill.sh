#!/bin/bash
# This is your script.sh

python3 -u -m curvemetrics.scripts.frontfill.raw_data > "./logs/frontfill/frontfill.log" 2>&1
python3 -u -m curvemetrics.scripts.frontfill.metrics >> "./logs/frontfill/frontfill.log" 2>&1
python3 -u -m curvemetrics.scripts.frontfill.takers >> "./logs/frontfill/frontfill.log" 2>&1

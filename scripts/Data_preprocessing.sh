#!/bin/bash

python Data_preprocessing/id_mapping.py

python Data_preprocessing/g_mask.py

python Data_preprocessing/g_color.py

python Data_preprocessing/show.py
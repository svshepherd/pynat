"""helper functions building on pyinaturalist"""

import requests
import pandas as pd
import datetime as dt
from typing import Tuple, Union
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import pyinaturalist as inat
import dill
import os

## load api key
with open('pyinaturalistkey.pkd', 'rb') as f:
    API_KEY = dill.load(f)

### lots left tbd
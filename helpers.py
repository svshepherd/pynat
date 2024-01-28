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
import ipyplot

## load api key
with open('pyinaturalistkey.pkd', 'rb') as f:
    API_KEY = dill.load(f)


def get_mine(uname:str, lookback_in_days: int, api_key: str=API_KEY) -> None:
    """
    Loads my observations by observation time (by iconic taxon?) in format appropriate for photo labels
    """
    # Define the base URL for the iNaturalist API
    base_url = "https://api.inaturalist.org/v1/observations"

    # Define the current date
    now = dt.datetime.now()
    start_date = now - dt.timedelta(days=lookback_in_days)

    response = inat.get_observations(user_id=[uname], d1=start_date, page='all')

    df = pd.json_normalize(response['results']) 

    for index, row in df.iloc[::-1].iterrows():
        print(f"""\n\n{row['observed_on']:%Y%m%d} {row['taxon.name']} ({row['species_guess']}) [inat obs id: {row['id']}]""")
        try:
            # response = requests.get(row.photos[0]['url'].replace('square','small'))
            # response.raise_for_status()  # Raise exception if the request failed
            # img = mpimg.imread(BytesIO(response.content), format='jpg')
            # plt.imshow(img)
            # plt.xticks([])  # Hide x tick labels
            # plt.yticks([])  # Hide y tick labels
            # plt.show()

            # if len(row.photos)>1:
            images = [each_obs_photo['url'].replace('square','small') for each_obs_photo in row.photos]
            ipyplot.plot_images(images)
            
        except requests.exceptions.RequestException as err:
            print(f"Failed to load images: {err}")


def coming_soon(kind:str,
                loc:tuple[float,float,float]=(37.6669, -77.8883, 25),
                norm:str=None,
                limit:int=10,
                ) -> None:
    """shows organism which have been observed in this season near here.
    
    roadmap:
    try to narrow time frame to consistently 21-day period?
    normalizations for sort order:
        count of all observations at time/place (by taxa / overall)
        count of all observations at time/place by taxa vs (10x loc radius, all-time)?
        count of all observations in time (10x loc radius?) and at place (longer window?) separately
        count of all observations in time and at place separately by taxa
        (similar but by phenology?)
        (separate totals for 'research grade' and 'informal' counts)

    photographs should match requested phenotype where possible
    add support for caterpillars/butterflies (and similar for benthic macroinverts?)
    split animals by clade and/or generalize interface?
    """
    assert norm==None, "not implemented yet"

    if kind == 'animals':
        taxa = {'taxon_id':1}
    # elif kind == 'plants':
    #     taxa = {'taxon_id':?}
    elif kind == 'flowers':
        taxa = {'term_id':12, 'term_value_id':13}
    elif kind == 'fruits':
        taxa = {'term_id':12, 'term_value_id':14}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    else:
        raise ValueError(f"kind '{kind}' not implemented")

    if isinstance(loc,tuple) & (len(loc)==3):
        place = {'lat':loc[0], 
                 'lng':loc[1], 
                 'radius':loc[2]}
    else:
        raise ValueError(f"expected loc triple of lat,long,radius")

    results = inat.get_observation_species_counts(
        month=list(set( [(dt.date.today()+dt.timedelta(days=-7)).month, (dt.date.today()+dt.timedelta(days=7)).month] )),
        verifiable=True,
        #per_page=0,
        **taxa,
        **place,
    )

    # Normalize results to DataFrame
    df_species_counts = pd.json_normalize(results['results'])

    ## todo: normalize each taxa by observation rate. numbers to consider
    # all observations in time/place
    # all observations in time for taxa and in place for taxa
    # all observations in time and in place separately
    # we want to sort things higher if more likely to be seen here&now relative to other times.
    # (ie if a species is ONLY seen in the blue ridge in may, we'd like it to be shown first!)

    # Display species names and their main images
    for index, row in df_species_counts.head(limit).iterrows():
        taxon_name = row['taxon.name']
        common_name = row.get('taxon.preferred_common_name', 'N/A')
        image_url = row['taxon.default_photo.medium_url']

        print(f"\n{taxon_name} ({common_name})")

        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise exception if the request failed
            img = mpimg.imread(BytesIO(response.content), format='jpg')
            plt.imshow(img)
            plt.xticks([])  # Hide x tick labels
            plt.yticks([])  # Hide y tick labels
            plt.show()
        except requests.exceptions.RequestException as e:
            print(f"Failed to load image: {e}")
        ### It'd be nice to specifically select images w/ appropriate phenotype
    
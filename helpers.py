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


def get_mine(uname:str, lookback_to: dt.datetime=None, lookback_in_days: int=None, api_key: str=API_KEY) -> None:
    """
    Loads my observations by observation time (by iconic taxon?) in format appropriate for photo labels
    """
    assert not (lookback_to and lookback_in_days), "only one of lookback_to or lookback_in_days should be provided"
    
    if not lookback_to and not lookback_in_days:
        print('assuming lookback_in_days=1')
        lookback_in_days = 1

    # Define the base URL for the iNaturalist API
    base_url = "https://api.inaturalist.org/v1/observations"

    # Define the current date
    now = dt.datetime.now()
    if lookback_to:
        start_date = lookback_to
    else:
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
                places:list[int]=None,
                loc:tuple[float,float,float]=None,
                norm:str=None,
                limit:int=10,
                ) -> None:
    """shows organism which have been observed in this season nearby, sorted by relative frequency.
   
    roadmap:
    turn names into links to iNaturalist
    try to narrow time frame to consistently 21-day period?
    photographs should match requested phenotype where possible
    add support for caterpillars/butterflies
    split animals by clade and/or generalize interface?
    """
    assert not (places and loc), "only one of places and loc should be provided"
    
    if not places and not loc:
        print('no place or location specified, assuming loc=(37.6669, -77.8883, 25)')
        loc = (37.6669, -77.8883, 25)

    assert norm in [None, 'time', 'place', 'overall'], "norm must be one of None, 'time', 'place', or 'overall'"

    if kind == 'animals':
        taxa = {'taxon_id':1}
    elif kind == 'plants':
        taxa = {'taxon_name':'plants'}
    elif kind == 'flowers':
        taxa = {'term_id':12, 'term_value_id':13}
    elif kind == 'fruits':
        taxa = {'term_id':12, 'term_value_id':14}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    else:
        raise ValueError(f"kind '{kind}' not implemented")

    if places:
        place = {'place_id':places}
    elif isinstance(loc,tuple) & (len(loc)==3):
        place = {'lat':loc[0], 
                 'lng':loc[1], 
                 'radius':loc[2]}
    else:
        raise ValueError(f"expected loc triple of lat,long,radius")

    time = {'month':list(set( [(dt.date.today()+dt.timedelta(days=-7)).month, (dt.date.today()+dt.timedelta(days=14)).month] ))}

    # ## fancier time resolution    
    # time = []
    # strt = dt.date.today()+dt.timedelta(days=-7)
    # fnsh = dt.date.today()+dt.timedelta(days=7)
    # dates = pd.date_range(start=strt, end=fnsh, freq='D')
    # for month in dates.month.unique():
    #     time.append({'month':month, 'day':list(dates[dates.month==month].day)})

    # results = []
    # for t in time:
    #     res = inat.get_observation_species_counts(
    #         verifiable=True,
    #         **taxa,
    #         **t,
    #         **place,
    #     )
    #     res = pd.json_normalize(res['results'])
    #     results.append(res)
    # results = pd.concat(results)
    ### ugh but need to merge these to sum counts and leave the other fields alone?
    ### also need to hack something similar for the 'place' normalizer

    results = inat.get_observation_species_counts(
        verifiable=True,
        **taxa,
        **time,
        **place,
    )
    results = pd.json_normalize(results['results'])

    if norm:
        results['normalizer'] = None
        for index, row in results.iterrows():
            taxon_id = row['taxon.id']
            ## if phenology is specified, I'd like to get species : phenology
            ## if phenology is not specified, I'd like to get iconic taxa?
            if norm == 'time':
                results.loc[index,'normalizer'] = inat.get_observations(taxon_id=taxon_id, **(taxa if 'term_value_id' in taxa.keys() else {}),
                                                                        **place, per_page=0)['total_results']
            if norm == 'place':
                results.loc[index,'normalizer'] = inat.get_observations(taxon_id=taxon_id, **(taxa if 'term_value_id' in taxa.keys() else {}), 
                                                                        **time, per_page=0)['total_results']
            if norm == 'overall':
                results.loc[index,'normalizer'] = inat.get_observations(taxon_id=taxon_id, **(taxa if 'term_value_id' in taxa.keys() else {}),
                                                                        per_page=0)['total_results']
        results['sorter'] = results['count']/results['normalizer']
        results.sort_values('sorter', ascending=False, inplace=True)

    # Display species names and their main images
    for index, row in results.head(limit).iterrows():
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
            
    return results
    
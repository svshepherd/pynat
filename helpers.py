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


def get_mine(uname:str, 
             lookback_in_days: int=None, 
             STRT: dt.datetime=dt.date.today(), 
             FNSH: dt.datetime=dt.date.today()+dt.timedelta(days=1), 
             api_key: str=API_KEY) -> None:
    """
    Loads my observations by observation time (by iconic taxon?) in format appropriate for photo labels

    uname: iNaturalist username
    lookback_to: datetime object
    lookback_in_days: int
    api_key: str
    """
    
    # Define the base URL for the iNaturalist API
    base_url = "https://api.inaturalist.org/v1/observations"

    # Define scope
    if lookback_in_days:
        start_date = STRT - dt.timedelta(days=lookback_in_days)
    else:
        start_date = STRT
    end_date = FNSH

    response = inat.get_observations(user_id=[uname], 
                                     d1=start_date, 
                                     d2=end_date,
                                     page='all')

    df = pd.json_normalize(response['results']).sort_values('observed_on')

    for index, row in df.iterrows():
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
    kind: 'any', 'plants', 'mushrooms', 'animals', 'wugs', 'fish', 'herps', 'birds', 'mammals'
    places: list of place ids [optional]
    loc: (lat, long, radius) [optional]
    norm: 'time', 'place', 'overall' [optional]
    limit: number of results to return [optional]
   
    roadmap:
    turn names into links to iNaturalist
    photographs should match requested phenotype where possible
    """
    assert not (places and loc), "only one of places and loc should be provided"
    
    if not places and not loc:
        print('no place or location specified, assuming loc=(37.6669, -77.8883, 25)')
        loc = (37.6669, -77.8883, 25)

    assert norm in [None, 'time', 'place', 'overall'], "norm must be one of None, 'time', 'place', or 'overall'"

    if kind == 'any':
        taxa = {}
    elif kind == 'plants':
        taxa = {'taxon_name':'plants'}
    elif kind == 'flowers':
        taxa = {'term_id':12, 'term_value_id':13}
    elif kind == 'fruits':
        taxa = {'term_id':12, 'term_value_id':14}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    elif kind == 'animals':
        taxa = {'taxon_id':1}
    elif kind == 'fish':
        taxa = {'taxon_id':47178}
    elif kind == 'mammals':
        taxa = {'taxon_id':40151}
    elif kind == 'birds':
        taxa = {'taxon_id':3}
    elif kind == 'herps':
        taxa = {'taxon_id':[26036, 20978]}
    elif kind == 'wugs':
        taxa = {'taxon_id':1, 'not_id': 355675}
    elif kind == 'butterflies':
        taxa = {'taxon_id': 47157, 'term_id':1, 'term_value_id':2}
    elif kind == 'caterpillars':
        taxa = {'taxon_id': 47157, 'term_id':1, 'term_value_id':6}
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

    time = []
    strt = dt.date.today()+dt.timedelta(days=-6)
    fnsh = dt.date.today()+dt.timedelta(days=7)
    dates = pd.date_range(start=strt, end=fnsh, freq='D')
    for month in dates.month.unique():
        time.append({'month':month, 'day':list(dates[dates.month==month].day)})

    COLS = ['taxon.id', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url', 'taxon.default_photo.medium_url', 'count']

    results = []
    for t in time:
        results.append(pd.json_normalize(inat.get_observation_species_counts(verifiable=True,
                                                                            **taxa,
                                                                            **t,
                                                                            **place,)['results']))
    results = pd.concat(results)[COLS]
    results = results.groupby(['taxon.id', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url', 'taxon.default_photo.medium_url']).sum().reset_index()

    if norm:
        if norm == 'place':
            normer = []
            for t in time:
                normer.append(pd.json_normalize(inat.get_observation_species_counts(taxon_id=results['taxon.id'].to_list(), 
                                                                                    **{key: value for key, value in taxa.items() if key == 'term_value_id'},
                                                                                    **t,
                                                                                    verifiable=True,)['results']))
            normer = pd.concat(normer)
            normer = normer.groupby('taxon.id')['count'].sum()
        else:
            normer = pd.json_normalize(inat.get_observation_species_counts(taxon_id=results['taxon.id'].to_list(), 
                                                                            **{key: value for key, value in taxa.items() if key == 'term_value_id'},
                                                                            **(place if norm=='time' else {}), 
                                                                            verifiable=True,)['results']).set_index('taxon.id')['count']
        results['normalizer'] = results['taxon.id'].map(normer)
        results['sorter'] = results['count']/results['normalizer']
        results.sort_values('sorter', ascending=False, inplace=True)

    # Display species names and their main images
    for index, row in results.head(limit).iterrows():
        taxon_name = row['taxon.name']
        common_name = row.get('taxon.preferred_common_name', 'N/A')
        image_url = row['taxon.default_photo.medium_url']

        print(f"\n{taxon_name} ({common_name}) - {row['taxon.wikipedia_url']}")

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


def get_park_data(geocenter:tuple, kind:str, limit:int) -> pd.DataFrame:
    """returns the most common species in a park, sorted by relative frequency.
    geocenter: (lat, long, radius)
    kind: 'any', 'plants', 'mushrooms', 'animals', 'wugs', 'fish', 'herps', 'birds', 'mammals'
    limit: number of results to return"""

    if kind == 'any':
        taxa = {}
    elif kind == 'plants':
        taxa = {'taxon_name':'plants'}
    elif kind == 'mushrooms':
        taxa = {'taxon_id':47170}
    elif kind == 'animals':
        taxa = {'taxon_id':1}
    elif kind == 'wugs':
        taxa = {'taxon_id':1, 'not_id': 355675}
    elif kind == 'fish':
        taxa = {'taxon_id':47178}
    elif kind == 'herps':
        taxa = {'taxon_id':[26036, 20978]}
    elif kind == 'birds':
        taxa = {'taxon_id':3}
    elif kind == 'mammals':
        taxa = {'taxon_id':40151}
    else:
        return "unknown kind"
    
    res = pd.json_normalize(inat.get_observation_species_counts(lat=geocenter[0], 
                                                                lng=geocenter[1], 
                                                                radius=geocenter[2], 
                                                                **taxa,
                                                                verifiable=True,)['results'])
    normer = pd.json_normalize(inat.get_observation_species_counts(taxon_id=res['taxon.id'].to_list(),
                                                                   verifiable=True,)['results'])
    res['normalizer'] = res['taxon.id'].map(normer.set_index('taxon.id')['count'])
    res['sorter'] = res['count']/res['normalizer']
    res.sort_values('sorter',ascending=False,inplace=True)
    print(f"{kind}:")
    if len(res)>499:
        print('too many results; normalization may be incomplete')
    return res[['count', 'taxon.name', 'taxon.preferred_common_name', 'taxon.wikipedia_url']].head(limit)
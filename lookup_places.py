import requests
import time

check_ids = [13446, 13435, 13434, 13451, 13483, 13430, 13445, 108239]
headers = {'Accept': 'application/json'}
print('=== Place details for independent city IDs ===')
for pid in check_ids:
    r = requests.get(f'https://api.inaturalist.org/v1/places/{pid}', headers=headers)
    results = r.json().get('results', [{}])
    d = results[0] if results else {}
    name = d.get('name', '?')
    display = d.get('display_name', '?')
    admin = d.get('admin_level')
    print(f'id={pid}: name={name!r} display={display!r} admin={admin}')
    time.sleep(0.3)

print()
print('=== Franklin County VA extended ===')
r = requests.get('https://api.inaturalist.org/v1/places/autocomplete', params={'q': 'Franklin County', 'per_page': 20}, headers=headers)
for p in r.json().get('results', []):
    dn = p.get('display_name', '')
    if 'VA' in dn:
        print(f'  id={p["id"]} name={dn}')

print()
print('=== Madison County VA extended ===')
r = requests.get('https://api.inaturalist.org/v1/places/autocomplete', params={'q': 'Madison County', 'per_page': 20}, headers=headers)
for p in r.json().get('results', []):
    dn = p.get('display_name', '')
    if 'VA' in dn:
        print(f'  id={p["id"]} name={dn}')

print()
print('=== Searches for remaining ===')
need_lookup = [
    ('Cumberland County', 'Cumberland County'),
    ('Franklin County', 'Franklin County'),
    ('Greene County', 'Greene County'),
    ('Henry County', 'Henry County'),
    ('Madison County', 'Madison County'),
    ('Orange County', 'Orange County'),
    ('Stafford County', 'Stafford County'),
    ('City of Alexandria', 'Alexandria'),
    ('City of Charlottesville', 'Charlottesville'),
    ('City of Danville', 'Danville'),
    ('City of Falls Church', 'Falls Church'),
    ('City of Lynchburg', 'Lynchburg'),
    ('City of Manassas Park', 'Manassas Park'),
    ('City of Martinsville', 'Martinsville'),
]

headers = {'Accept': 'application/json'}
for label, query in need_lookup:
    r = requests.get(
        'https://api.inaturalist.org/v1/places/autocomplete',
        params={'q': query, 'per_page': 15},
        headers=headers,
    )
    places = r.json().get('results', [])
    print(f'{label}:')
    for p in places[:10]:
        pid = p['id']
        adm = p.get('admin_level')
        name = p.get('display_name') or p.get('name', '')
        print(f'  id={pid} admin_level={adm} name={name}')
    print()
    time.sleep(0.4)

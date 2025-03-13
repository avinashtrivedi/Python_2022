from datetime import datetime

import requests
from requests.structures import CaseInsensitiveDict
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

#####################################

import yaml
yaml_file = 'config.yaml'

with open(yaml_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

api_key = config['API_KEYS']['key']
#####################################

#
def get_travel_time_polygon(lat, lon, travel_time):
    """
    lat: float latitude
    lon: float longitude
    travel_time: int seconds
    """
    url = "https://api.traveltimeapp.com/v4/time-map"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "application/geo+json"
    headers["X-Application-Id"] = "24d60183"
    headers["X-Api-Key"] = api_key
    data = """
    {
      "departure_searches": [
        {
          "id": "GeoJSON_Test",
          "coords": {
            "lat": %f,
            "lng": %f
          },
          "transportation": {
            "type": "driving"
          },
          "departure_time": "%s",
          "travel_time": %d
          }
      ]}
    """ % (lat, lon, now, travel_time)

    resp = requests.post(url, headers=headers, data=data)
    return resp.json()


# def get_travel_time_polygon_v2(lat, lon, sec, m):
#     data = {"locations": [[lon, lat]],
#             "range": [sec, m]
#             }
#
#     headers = {
#         'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
#         'Authorization': '5b3ce3597851110001cf624853d3c5c996914e49a3ce4c388a39116b',
#         'Content-Type': 'application/json; charset=utf-8'
#     }
#     resp = requests.post('https://api.openrouteservice.org/v2/isochrones/driving-car', json=data, headers=headers)
#
#     # print(resp.status_code, resp.reason)
#     # print(resp.text)
#     v2 = resp.json()
#     coordinates = [v2['features'][0]['geometry']['coordinates']]
#     resp = {'features': [v2['features'][0]],
#             'type': v2['type']}
#     resp['features'][0]['geometry']['coordinates'] = coordinates
#
#     return resp

# def get_travel_time_polygon_v2(lat, lon, travel_time):
#     data = {"locations": [[lon, lat]],
#             "range": [travel_time]
#             }
#
#     headers = {
#         'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
#         'Authorization': '5b3ce3597851110001cf624853d3c5c996914e49a3ce4c388a39116b',
#         'Content-Type': 'application/json; charset=utf-8'
#     }
#     resp = requests.post('https://api.openrouteservice.org/v2/isochrones/driving-car', json=data, headers=headers)
#
#     # print(resp.status_code, resp.reason)
#     # print(resp.text)
#
#     # Convert json response to match api.traveltimeapp
#     v2 = resp.json()
#     coordinates = [v2['features'][0]['geometry']['coordinates']]
#     resp = {'features': [v2['features'][0]],
#             'type': v2['type']}
#     resp['features'][0]['geometry']['coordinates'] = coordinates
#
#     return resp
#
def get_travel_time_polygon_v2(lat, lon, travel_time):
    """
        lat: float latitude
        lon: float longitude
        travel_time: int seconds
        """
    data = {"locations": [[lon, lat]],
            "range": [travel_time],
            "intersections": "true"
            }

    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    resp = requests.post('https://api.openrouteservice.org/v2/isochrones/driving-car', json=data, headers=headers)

    # print(resp.status_code, resp.reason)
    # print(resp.text)

    # Convert json response to match api.traveltimeapp
    v2 = resp.json()
    coordinates = [v2['features'][0]['geometry']['coordinates']]
    resp = {'features': [v2['features'][0]],
            'type': v2['type']}
    resp['features'][0]['geometry']['coordinates'] = coordinates
    resp['features'][0]['geometry']['type'] = 'MultiPolygon'
    resp['features'][0]['properties'] = {}

    return resp


print(get_travel_time_polygon_v2(27.960154, -26.050087, 300))


def close_holes(poly: Polygon) -> Polygon:
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def merge_multi_poly(multi_poly: MultiPolygon) -> Polygon:
    polies = [close_holes(poly) for poly in multi_poly.geoms]

    return unary_union(polies)


def select_largest(multi_poly: MultiPolygon) -> Polygon:
    return max(
        [poly for poly in multi_poly.geoms],
        key=lambda x: x.area,
    )


def simplify(multi_poly: MultiPolygon) -> Polygon:
    return close_holes(select_largest(multi_poly))


def convex_hull(multi_poly: MultiPolygon) -> Polygon:
    return multi_poly.convex_hull


def down_sample(poly: Polygon, step=2) -> Polygon:
    coords = poly.exterior.coords
    return Polygon(
        shell=[c for c in coords][::step]
    )

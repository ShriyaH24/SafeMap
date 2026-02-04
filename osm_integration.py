import requests
import json
from OSMPythonTools.overpass import Overpass
import time

class OSMIntegration:
    def __init__(self):
        self.overpass = Overpass()
        self.city_coordinates = {
            'Delhi City': {'lat': 28.6139, 'lon': 77.2090},
            'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
            'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
            # Add more cities as needed
        }
    
    def get_city_safety_data(self, city_name, radius_km=5):
        """Fetch safety infrastructure from OpenStreetMap"""
        if city_name not in self.city_coordinates:
            return None
        
        coords = self.city_coordinates[city_name]
        
        # Overpass query for safety features
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="police"](around:{radius_km*1000},{coords['lat']},{coords['lon']});
          node["highway"="street_lamp"](around:{radius_km*1000},{coords['lat']},{coords['lon']});
          node["man_made"="surveillance"](around:{radius_km*1000},{coords['lat']},{coords['lon']});
          node["amenity"="emergency_phone"](around:{radius_km*1000},{coords['lat']},{coords['lon']});
        );
        out body;
        """
        
        try:
            result = self.overpass.query(query)
            return self._process_osm_result(result)
        except Exception as e:
            print(f"OSM Query failed: {e}")
            return self._get_mock_data(city_name)
    
    def _process_osm_result(self, result):
        """Process OSM query result"""
        data = {
            'police_stations': [],
            'street_lights': [],
            'cctv_cameras': [],
            'emergency_phones': []
        }
        
        for element in result.elements():
            if 'amenity' in element.tags():
                if element.tags()['amenity'] == 'police':
                    data['police_stations'].append({
                        'lat': element.lat(),
                        'lon': element.lon(),
                        'name': element.tags().get('name', 'Police Station')
                    })
                elif element.tags()['amenity'] == 'emergency_phone':
                    data['emergency_phones'].append({
                        'lat': element.lat(),
                        'lon': element.lon()
                    })
            
            elif 'highway' in element.tags() and element.tags()['highway'] == 'street_lamp':
                data['street_lights'].append({
                    'lat': element.lat(),
                    'lon': element.lon()
                })
            
            elif 'man_made' in element.tags() and element.tags()['man_made'] == 'surveillance':
                data['cctv_cameras'].append({
                    'lat': element.lat(),
                    'lon': element.lon()
                })
        
        return data
    
    def _get_mock_data(self, city_name):
        """Return mock data if OSM fails (for demo)"""
        return {
            'police_stations': [
                {'lat': 12.9716 + random.uniform(-0.05, 0.05), 
                 'lon': 77.5946 + random.uniform(-0.05, 0.05),
                 'name': 'Central Police Station'}
                for _ in range(random.randint(5, 15))
            ],
            'street_lights': [
                {'lat': 12.9716 + random.uniform(-0.1, 0.1),
                 'lon': 77.5946 + random.uniform(-0.1, 0.1)}
                for _ in range(random.randint(50, 200))
            ],
            'cctv_cameras': [
                {'lat': 12.9716 + random.uniform(-0.08, 0.08),
                 'lon': 77.5946 + random.uniform(-0.08, 0.08)}
                for _ in range(random.randint(20, 80))
            ],
            'emergency_phones': [
                {'lat': 12.9716 + random.uniform(-0.06, 0.06),
                 'lon': 77.5946 + random.uniform(-0.06, 0.06)}
                for _ in range(random.randint(3, 10))
            ]
        }
    
    def get_route_safety(self, route_coordinates, time_of_day):
        """Calculate safety along a route"""
        # This would analyze route segments against OSM data
        # For demo, return mock safety scores
        
        safety_scores = []
        for i, coord in enumerate(route_coordinates):
            # Simulate varying safety along route
            base_score = 70 + 20 * np.sin(i/10)
            if time_of_day in ['Night', 'Evening']:
                base_score -= 15
            
            safety_scores.append({
                'lat': coord[0],
                'lon': coord[1],
                'safety_score': max(0, min(100, base_score + random.uniform(-5, 5))),
                'infrastructure': random.choice(['Good', 'Moderate', 'Poor'])
            })
        
        return safety_scores
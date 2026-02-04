import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_map(center_coords, city_name, data, show_police=True, show_lights=False, route=None):
    """Create interactive Folium map"""
    m = folium.Map(location=center_coords, zoom_start=13, control_scale=True)
    
    # Add tile layers
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('cartodbpositron').add_to(m)  # Light mode
    
    # Add city boundary (simulated)
    folium.Circle(
        location=center_coords,
        radius=2000,
        color='#3186cc',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.1,
        popup=f"{city_name} Safety Zone"
    ).add_to(m)
    
    # Add crime heatmap (simulated)
    heat_data = []
    for _ in range(50):
        lat = center_coords[0] + np.random.uniform(-0.05, 0.05)
        lon = center_coords[1] + np.random.uniform(-0.05, 0.05)
        weight = np.random.uniform(0.1, 1.0)
        heat_data.append([lat, lon, weight])
    
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Add route if provided
    if route:
        route_points = [
            [center_coords[0] - 0.01, center_coords[1] - 0.01],
            [center_coords[0] - 0.005, center_coords[1]],
            [center_coords[0], center_coords[1] + 0.01],
            [center_coords[0] + 0.01, center_coords[1] + 0.005]
        ]
        
        folium.PolyLine(
            route_points,
            weight=4,
            color='#3B82F6',
            opacity=0.8,
            popup='Safe Route'
        ).add_to(m)
        
        # Add route markers
        folium.Marker(
            route_points[0],
            popup='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            route_points[-1],
            popup='Destination',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    from folium.plugins import MiniMap
    minimap = MiniMap()
    m.add_child(minimap)
    
    # Add fullscreen button
    from folium.plugins import Fullscreen
    Fullscreen().add_to(m)
    
    return m

def create_safety_chart(city_data):
    """Create safety comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=city_data['City_Name'],
        y=city_data['safemap_risk'] * 100,
        name='Crime Risk %',
        marker_color='indianred'
    ))
    
    fig.add_trace(go.Bar(
        x=city_data['City_Name'],
        y=city_data['severity_score'] / 10,  # Scale down for visualization
        name='Severity Score',
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title='City Safety Comparison',
        xaxis_title='City',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def create_time_series_chart(time_data):
    """Create time-based safety chart"""
    fig = px.area(time_data, x='Hour', y='Risk_Score',
                  title='24-Hour Safety Pattern',
                  labels={'Risk_Score': 'Safety Score (Higher is safer)'})
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 3)),
            ticktext=[f'{h:02d}:00' for h in range(0, 24, 3)]
        )
    )
    
    return fig
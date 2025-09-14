"""
Austin, TX Telematics Data Generator for Insurance POC
Enhanced version with Snowflake integration, data quality validation, and comprehensive analytics
Updated: Removed gamification, added driver profiles and incidents tables
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import random
from geopy.distance import geodesic
import sqlite3
from typing import List, Tuple, Dict, Optional
import time
import os
import sys
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Snowflake connector not available. Will use SQLite for local storage.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telematics_generator.log', mode='a'),
        logging.StreamHandler()
    ]
)

class SnowflakeConfig:
    """Snowflake configuration management"""
    
    @staticmethod
    def get_connection_params():
        """Get Snowflake connection parameters from environment variables"""
        return {
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE', 'AUSTIN_TELEMATICS'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'RAW_DATA'),
            'role': os.getenv('SNOWFLAKE_ROLE', 'SYSADMIN'),
            'client_session_keep_alive': True,
            'connection_timeout': 60
        }

class AustinTelematicsGenerator:
    def __init__(self, use_snowflake: bool = True):
        """
        Initialize the Austin Telematics Data Generator
        
        Args:
            use_snowflake (bool): Whether to use Snowflake for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.use_snowflake = use_snowflake and SNOWFLAKE_AVAILABLE
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Austin, TX geographical boundaries
        self.austin_bounds = {
            'north': 30.5168,  # North Austin
            'south': 30.0986,  # South Austin
            'east': -97.5684,  # East Austin
            'west': -97.9383   # West Austin
        }
        
        # Major Austin locations for realistic trip patterns
        self.major_locations = [
            {'name': 'Downtown Austin', 'lat': 30.2672, 'lon': -97.7431},
            {'name': 'UT Campus', 'lat': 30.2849, 'lon': -97.7341},
            {'name': 'South Lamar', 'lat': 30.2467, 'lon': -97.7697},
            {'name': 'North Austin', 'lat': 30.4518, 'lon': -97.7341},
            {'name': 'East Austin', 'lat': 30.2672, 'lon': -97.7031},
            {'name': 'Westlake', 'lat': 30.3077, 'lon': -97.8081},
            {'name': 'Round Rock', 'lat': 30.5083, 'lon': -97.6789},
            {'name': 'Cedar Park', 'lat': 30.5052, 'lon': -97.8203},
            {'name': 'Pflugerville', 'lat': 30.4394, 'lon': -97.6200},
            {'name': 'Lakeway', 'lat': 30.3688, 'lon': -97.9747}
        ]
        
        # Initialize data storage
        self.incident_data = []
        self.driver_profiles = []
        
        # Load incident data
        self.load_incident_data()
        
        # Log initialization
        if self.use_snowflake:
            self.logger.info("Snowflake integration enabled")
        else:
            self.logger.info("ℹUsing SQLite for local storage")
        
    def load_incident_data(self):
        """Load real-time traffic incident data from Austin Open Data"""
        try:
            self.logger.info("Loading Austin traffic incident data...")
            url = "https://data.austintexas.gov/api/views/dx9v-zd7x/rows.json?accessType=DOWNLOAD"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                columns = [col['name'] for col in data['meta']['view']['columns']]
                
                for row in data['data']:
                    try:
                        incident = dict(zip(columns, row))
                        
                        # Extract coordinates if available
                        if incident.get('Location') and 'POINT' in str(incident['Location']):
                            coords = str(incident['Location']).replace('POINT (', '').replace(')', '').split()
                            if len(coords) == 2:
                                incident['parsed_lon'] = float(coords[0])
                                incident['parsed_lat'] = float(coords[1])
                                incident['fetched_at'] = datetime.now()
                                
                                # Only include incidents within Austin bounds
                                if (self.austin_bounds['south'] <= incident['parsed_lat'] <= self.austin_bounds['north'] and
                                    self.austin_bounds['west'] <= incident['parsed_lon'] <= self.austin_bounds['east']):
                                    
                                    # Clean and standardize incident data
                                    cleaned_incident = {
                                        'INCIDENT_ID': incident.get('Traffic Report ID', f"INCIDENT_{len(self.incident_data)}"),
                                        'ISSUE_REPORTED': incident.get('Issue Reported', 'Unknown'),
                                        'PUBLISHED_DATE': incident.get('Published Date'),
                                        'LOCATION_DESCRIPTION': incident.get('Address', 'Unknown Location'),
                                        'LATITUDE': incident['parsed_lat'],
                                        'LONGITUDE': incident['parsed_lon'],
                                        'STATUS': incident.get('Status', 'Unknown'),
                                        'AGENCY': incident.get('Agency', 'Unknown'),
                                        'FETCHED_AT': incident['fetched_at']
                                    }
                                    self.incident_data.append(cleaned_incident)
                    except Exception as e:
                        continue
                        
            self.logger.info(f"Loaded {len(self.incident_data)} traffic incidents from Austin Open Data")
            
        except Exception as e:
            self.logger.warning(f"Could not load incident data: {e}")
            self.logger.info("Proceeding with synthetic data generation without real incidents")
    
    def generate_driver_profiles(self, num_drivers: int) -> List[Dict]:
        """Generate diverse driver profiles with different risk characteristics"""
        
        self.logger.info(f"Generating {num_drivers} driver profiles...")
        
        driver_types = ['safe', 'average', 'aggressive', 'elderly', 'young']
        profiles = []
        
        for i in range(num_drivers):
            driver_type = random.choice(driver_types)
            
            # Base profile
            profile = {
                'driver_id': f'DRV_{i:04d}',
                'driver_type': driver_type,
                'age': self._generate_age_by_type(driver_type),
                'home_location': random.choice(self.major_locations)['name'],
                'work_location': random.choice(self.major_locations)['name'],
                'vehicle_year': random.randint(2015, 2024),
                'vehicle_make': random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Tesla']),
                'annual_mileage_estimate': random.randint(8000, 50000),
                'created_at': datetime.now(),
            }
            
            # Type-specific behavioral characteristics
            behavioral_params = {
                'safe': {
                    'base_speed_multiplier': 0.85,
                    'hard_braking_prob': 0.05,
                    'hard_accel_prob': 0.03,
                    'night_driving_prob': 0.1,
                    'weekend_driving_mult': 0.7
                },
                'aggressive': {
                    'base_speed_multiplier': 1.15,
                    'hard_braking_prob': 0.25,
                    'hard_accel_prob': 0.20,
                    'night_driving_prob': 0.3,
                    'weekend_driving_mult': 1.3
                },
                'elderly': {
                    'base_speed_multiplier': 0.90,
                    'hard_braking_prob': 0.15,
                    'hard_accel_prob': 0.08,
                    'night_driving_prob': 0.05,
                    'weekend_driving_mult': 0.8
                },
                'young': {
                    'base_speed_multiplier': 1.05,
                    'hard_braking_prob': 0.18,
                    'hard_accel_prob': 0.15,
                    'night_driving_prob': 0.25,
                    'weekend_driving_mult': 1.4
                },
                'average': {
                    'base_speed_multiplier': 1.0,
                    'hard_braking_prob': 0.12,
                    'hard_accel_prob': 0.10,
                    'night_driving_prob': 0.15,
                    'weekend_driving_mult': 1.0
                }
            }
            
            # Add behavioral parameters to profile
            profile.update(behavioral_params[driver_type])
            profiles.append(profile)
        
        self.driver_profiles = profiles
        return profiles
    
    def _generate_age_by_type(self, driver_type: str) -> int:
        """Generate realistic age based on driver type"""
        age_ranges = {
            'young': (18, 25),
            'elderly': (65, 85),
            'safe': (30, 55),
            'aggressive': (25, 45),
            'average': (25, 65)
        }
        
        min_age, max_age = age_ranges[driver_type]
        return random.randint(min_age, max_age)
    
    def generate_trip(self, driver_profile: Dict, start_time: datetime) -> List[Dict]:
        """Generate a single trip with realistic telematics data"""
        
        # Determine trip type and destinations
        trip_type = self._determine_trip_type(start_time, driver_profile)
        start_loc, end_loc = self._get_trip_locations(driver_profile, trip_type)
        
        # Calculate trip parameters
        distance = geodesic((start_loc['lat'], start_loc['lon']), 
                          (end_loc['lat'], end_loc['lon'])).kilometers
        
        # Estimate trip duration (including traffic)
        base_speed = 45 * driver_profile['base_speed_multiplier']
        trip_duration_minutes = max(10, (distance / base_speed) * 60 * 1.3)  # Add traffic buffer
        
        # Generate data points every 30 seconds
        num_points = max(5, min(int(trip_duration_minutes * 2), 50))  # Cap at 50 points
        trip_data = []
        
        current_time = start_time
        trip_id = f"{driver_profile['driver_id']}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        for i in range(num_points):
            # Calculate position along route (linear interpolation)
            progress = i / (num_points - 1)
            current_lat = start_loc['lat'] + (end_loc['lat'] - start_loc['lat']) * progress
            current_lon = start_loc['lon'] + (end_loc['lon'] - start_loc['lon']) * progress
            
            # Add realistic GPS noise
            current_lat += random.gauss(0, 0.0001)
            current_lon += random.gauss(0, 0.0001)
            
            # Generate speed data
            speed = self._generate_speed(progress, driver_profile, current_time, current_lat, current_lon)
            
            # Generate acceleration data
            if i == 0:
                acceleration = random.uniform(0.5, 2.0)
                prev_speed = 0
            else:
                prev_speed = trip_data[-1]['speed_kmh']
                speed_change = speed - prev_speed
                acceleration = (speed_change * 1000/3600) / 30
            
            # Check for incidents near current location
            incident_risk = self._check_incident_risk(current_lat, current_lon)
            
            # Generate events based on driver profile and conditions
            hard_braking = (random.random() < driver_profile['hard_braking_prob'] * incident_risk)
            hard_acceleration = (random.random() < driver_profile['hard_accel_prob'])
            
            if hard_braking:
                acceleration = random.uniform(-8.0, -4.0)
                speed = max(0, speed - 15)
            elif hard_acceleration and speed < 80:
                acceleration = random.uniform(3.0, 6.0)
            
            # Phone usage based on driver type
            phone_usage_probs = {'young': 0.15, 'aggressive': 0.15, 'average': 0.08}
            phone_usage = random.random() < phone_usage_probs.get(driver_profile['driver_type'], 0.03)
            
            # Weather conditions (simplified)
            weather_conditions = ['clear', 'rain', 'fog', 'cloudy']
            weather_weights = [0.7, 0.15, 0.05, 0.1]
            weather_condition = random.choices(weather_conditions, weights=weather_weights)[0]
            
            # Apply weather impact on speed
            weather_multipliers = {'clear': 1.0, 'rain': 0.85, 'fog': 0.7, 'cloudy': 0.95}
            speed *= weather_multipliers[weather_condition]
            
            trip_point = {
                'trip_id': trip_id,
                'driver_id': driver_profile['driver_id'],
                'timestamp': current_time,
                'latitude': round(current_lat, 6),
                'longitude': round(current_lon, 6),
                'speed_kmh': round(max(0, speed), 2),
                'acceleration_ms2': round(acceleration, 3),
                'hard_braking': hard_braking,
                'hard_acceleration': hard_acceleration,
                'phone_usage': phone_usage,
                'time_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_weekend': current_time.weekday() >= 5,
                'trip_type': trip_type,
                'weather_condition': weather_condition,
                'distance_from_home': round(geodesic((current_lat, current_lon), 
                                                   (start_loc['lat'], start_loc['lon'])).kilometers, 2),
                'trip_distance_km': round(distance, 2),
                'incident_risk_factor': round(incident_risk, 3),
                'start_location': start_loc['name'],
                'end_location': end_loc['name'],
                'created_at': datetime.now(),
                'data_quality_score': 1.0  # Will be calculated later
            }
            
            trip_data.append(trip_point)
            current_time += timedelta(seconds=30)
        
        return trip_data
    
    def _determine_trip_type(self, start_time: datetime, driver_profile: Dict) -> str:
        """Determine trip type based on time and driver characteristics"""
        hour = start_time.hour
        is_weekend = start_time.weekday() >= 5
        
        if is_weekend:
            return random.choice(['leisure', 'shopping', 'social', 'errands'])
        elif 7 <= hour <= 9:
            return 'commute_to_work'
        elif 17 <= hour <= 19:
            return 'commute_from_work'
        elif 12 <= hour <= 14:
            return 'lunch'
        else:
            return random.choice(['errands', 'shopping', 'social', 'personal'])
    
    def _get_trip_locations(self, driver_profile: Dict, trip_type: str) -> Tuple[Dict, Dict]:
        """Get start and end locations based on trip type"""
        
        # Find home and work locations from profile
        home_loc = next(loc for loc in self.major_locations if loc['name'] == driver_profile['home_location'])
        work_loc = next(loc for loc in self.major_locations if loc['name'] == driver_profile['work_location'])
        
        if trip_type == 'commute_to_work':
            return home_loc, work_loc
        elif trip_type == 'commute_from_work':
            return work_loc, home_loc
        else:
            # For other trips, start from home/work and go to random location
            start_options = [home_loc, work_loc]
            start_loc = random.choice(start_options)
            
            # Choose different end location
            available_locs = [loc for loc in self.major_locations if loc['name'] != start_loc['name']]
            end_loc = random.choice(available_locs)
            
            return start_loc, end_loc
    
    def _generate_speed(self, progress: float, driver_profile: Dict, 
                       current_time: datetime, lat: float, lon: float) -> float:
        """Generate realistic speed based on various factors"""
        
        # Base speed depends on location and time
        base_speed = 45  # km/h average city driving
        
        # Rush hour adjustments
        if current_time.weekday() < 5:  # Weekdays
            if 7 <= current_time.hour <= 9 or 17 <= current_time.hour <= 19:
                base_speed *= 0.7  # Slower in rush hour
        
        # Driver type adjustment
        speed = base_speed * driver_profile['base_speed_multiplier']
        
        # Trip progress adjustment (slower at start/end)
        if progress < 0.1 or progress > 0.9:
            speed *= 0.6
        elif 0.3 <= progress <= 0.7:
            speed *= 1.2
        
        # Add randomness
        speed += random.gauss(0, 8)
        
        # Ensure reasonable bounds
        return max(0, min(speed, 120))
    
    def _check_incident_risk(self, lat: float, lon: float) -> float:
        """Calculate incident risk based on proximity to real Austin incidents"""
        risk_factor = 1.0
        
        for incident in self.incident_data[:30]:  # Check recent incidents
            if 'latitude' in incident and 'longitude' in incident:
                distance = geodesic((lat, lon), 
                                  (incident['latitude'], incident['longitude'])).kilometers
                
                if distance < 2.0:  # Within 2km of incident
                    risk_factor += 0.5
                    if distance < 0.5:  # Very close to incident
                        risk_factor += 1.0
        
        return min(risk_factor, 3.0)  # Cap at 3x normal risk
    
    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality validation and scoring"""
        
        self.logger.info("🔍 Performing data quality validation...")
        
        initial_count = len(df)
        
        # Calculate data quality score for each record
        df['data_quality_score'] = 1.0
        
        # Reduce score for anomalies
        df.loc[df['speed_kmh'] > 150, 'data_quality_score'] -= 0.2
        df.loc[df['acceleration_ms2'].abs() > 8, 'data_quality_score'] -= 0.1
        df.loc[(df['latitude'] < self.austin_bounds['south']) | 
               (df['latitude'] > self.austin_bounds['north']), 'data_quality_score'] -= 0.3
        df.loc[(df['longitude'] < self.austin_bounds['west']) | 
               (df['longitude'] > self.austin_bounds['east']), 'data_quality_score'] -= 0.3
        
        # Remove very low quality records
        df = df[df['data_quality_score'] >= 0.5].copy()
        
        final_count = len(df)
        self.logger.info(f"✅ Data quality validation complete: {initial_count:,} → {final_count:,} records")
        
        return df
    
    def generate_synthetic_dataset(self, num_drivers: int, days: int) -> pd.DataFrame:
        """Generate complete synthetic telematics dataset"""
        
        self.logger.info(f"🚀 Generating telematics data for {num_drivers} drivers over {days} days...")
        
        # Generate driver profiles
        driver_profiles = self.generate_driver_profiles(num_drivers)
        
        all_trip_data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate trips for each driver
            for driver_profile in driver_profiles:
                # Determine number of trips for this driver on this day
                is_weekend = current_date.weekday() >= 5
                
                if is_weekend:
                    num_trips = random.choices([0, 1, 2, 3, 4], 
                                             weights=[10, 20, 30, 25, 15])[0]
                    num_trips = int(num_trips * driver_profile['weekend_driving_mult'])
                else:
                    num_trips = random.choices([1, 2, 3, 4, 5], 
                                             weights=[5, 25, 35, 25, 10])[0]
                
                # Generate trips
                for trip_num in range(num_trips):
                    # Realistic trip timing
                    if not is_weekend and trip_num == 0:
                        start_hour = random.randint(7, 9)  # Morning commute
                    elif not is_weekend and trip_num == 1:
                        start_hour = random.randint(17, 19)  # Evening commute
                    else:
                        start_hour = random.randint(8, 22)  # Other trips
                    
                    start_minute = random.randint(0, 59)
                    trip_start = current_date.replace(hour=start_hour, minute=start_minute)
                    
                    # Generate trip data
                    trip_data = self.generate_trip(driver_profile, trip_start)
                    all_trip_data.extend(trip_data)
            
            if day % 10 == 0:
                self.logger.info(f"📅 Completed day {day+1}/{days}")
        
        # Convert to DataFrame and validate
        df = pd.DataFrame(all_trip_data)
        df = self.validate_data_quality(df)
        
        self.logger.info(f"✅ Dataset generation complete:")
        self.logger.info(f"   📊 Total records: {len(df):,}")
        self.logger.info(f"   🚗 Unique drivers: {df['driver_id'].nunique()}")
        self.logger.info(f"   🛣️  Unique trips: {df['trip_id'].nunique()}")
        self.logger.info(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def _create_snowflake_tables(self, conn):
        """Create all necessary Snowflake tables"""
        cursor = conn.cursor()
        
        try:
            # Create telematics data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS TELEMATICS_DATA (
                    trip_id STRING,
                    driver_id STRING,
                    timestamp VARCHAR,
                    latitude FLOAT,
                    longitude FLOAT,
                    speed_kmh FLOAT,
                    acceleration_ms2 FLOAT,
                    hard_braking BOOLEAN,
                    hard_acceleration BOOLEAN,
                    phone_usage BOOLEAN,
                    time_of_day INT,
                    day_of_week INT,
                    is_weekend BOOLEAN,
                    trip_type STRING,
                    weather_condition STRING,
                    distance_from_home FLOAT,
                    trip_distance_km FLOAT,
                    incident_risk_factor FLOAT,
                    start_location STRING,
                    end_location STRING,
                    created_at VARCHAR,
                    data_quality_score FLOAT,
                    processing_batch STRING DEFAULT 'SYNTHETIC_BATCH_1'
                )
            """)
            
            # Create driver profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS DRIVER_PROFILES (
                    driver_id STRING PRIMARY KEY,
                    driver_type STRING,
                    age INT,
                    vehicle_make STRING,
                    vehicle_year INT,
                    annual_mileage_estimate INT,
                    home_location STRING,
                    work_location STRING,
                    base_speed_multiplier FLOAT,
                    hard_braking_prob FLOAT,
                    hard_accel_prob FLOAT,
                    night_driving_prob FLOAT,
                    weekend_driving_mult FLOAT,
                    created_at VARCHAR
                )
            """)
            
            # Create Austin incidents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS AUSTIN_INCIDENTS (
                    INCIDENT_ID STRING,
                    issue_reported STRING,
                    published_date STRING,
                    location_description STRING,
                    latitude FLOAT,
                    longitude FLOAT,
                    status STRING,
                    agency STRING,
                    fetched_at VARCHAR
                )
            """)
            
            self.logger.info("✅ Snowflake tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating Snowflake tables: {e}")
            raise
        finally:
            cursor.close()
    
    def save_to_snowflake(self, df: pd.DataFrame) -> bool:
        """Save all data to Snowflake with comprehensive error handling"""
        if not self.use_snowflake:
            self.logger.warning("Snowflake not available, skipping upload")
            return False
            
        try:
            self.logger.info("Connecting to Snowflake...")
            config = SnowflakeConfig()
            conn = snowflake.connector.connect(**config.get_connection_params())
            
            # Create tables
            self._create_snowflake_tables(conn)
            
            # Save telematics data
            self.logger.info("Uploading telematics data...")

            # Ensure columns match Snowflake's (unquoted) uppercase identifiers
            # Prepare DataFrame copy for upload
            df_to_load = df.copy()

            # Columns we want as strings in Snowflake
            timestamp_like_cols = ['timestamp', 'created_at', 'fetched_at']

            # Convert each timestamp-like column to ISO string 'YYYY-MM-DD HH:MM:SS'
            for col in timestamp_like_cols:
                # handle either lowercase or uppercase column names (incoming df may vary)
                if col in df_to_load.columns:
                    df_to_load[col] = pd.to_datetime(df_to_load[col], errors='coerce')
                    df_to_load[col] = df_to_load[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df_to_load[col] = df_to_load[col].fillna('')  # empty string for missing
                elif col.upper() in df_to_load.columns:
                    cu = col.upper()
                    df_to_load[cu] = pd.to_datetime(df_to_load[cu], errors='coerce')
                    df_to_load[cu] = df_to_load[cu].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df_to_load[cu] = df_to_load[cu].fillna('')

            # Uppercase column names to align with Snowflake created table
            df_to_load.columns = [c.upper() for c in df_to_load.columns]

            success1, _, nrows1, _ = write_pandas(
                conn, df_to_load, 'TELEMATICS_DATA', auto_create_table=False, overwrite=True
            )
            
            # Save driver profiles
            if self.driver_profiles:
                self.logger.info("Uploading driver profiles...")
                profiles_df = pd.DataFrame(self.driver_profiles)
                # convert created_at if present
                if 'created_at' in profiles_df.columns:
                    profiles_df['created_at'] = pd.to_datetime(profiles_df['created_at'], errors='coerce')\
                                                .dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                profiles_df.columns = [c.upper() for c in profiles_df.columns]
                success2, _, nrows2, _ = write_pandas(
                    conn, profiles_df, 'DRIVER_PROFILES', auto_create_table=False, overwrite=True
                )
            else:
                success2, nrows2 = True, 0

            
            # Save Austin incidents
            if self.incident_data:
                self.logger.info("Uploading Austin incidents...")
                incidents_df = pd.DataFrame(self.incident_data)
                # Convert fetched_at if present (could be uppercase already)
                if 'fetched_at' in incidents_df.columns:
                    incidents_df['fetched_at'] = pd.to_datetime(incidents_df['fetched_at'], errors='coerce')\
                                                .dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                if 'FETCHED_AT' in incidents_df.columns and 'fetched_at' not in incidents_df.columns:
                    incidents_df['FETCHED_AT'] = pd.to_datetime(incidents_df['FETCHED_AT'], errors='coerce')\
                                                .dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                incidents_df.columns = [c.upper() for c in incidents_df.columns]
                success3, _, nrows3, _ = write_pandas(
                    conn, incidents_df, 'AUSTIN_INCIDENTS', auto_create_table=False, overwrite=True
                )
            else:
                success3, nrows3 = True, 0
            
            conn.close()
            
            if success1 and success2 and success3:
                self.logger.info(f"Successfully saved to Snowflake:")
                self.logger.info(f"Telematics records: {nrows1:,}")
                self.logger.info(f"Driver profiles: {nrows2}")
                self.logger.info(f"Austin incidents: {nrows3}")
                return True
            else:
                self.logger.error("Some Snowflake operations failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving to Snowflake: {e}")
            return False
    
    def save_to_database(self, df: pd.DataFrame, db_path: str = 'data/austin_telematics.db'):
        """Save data to SQLite database as fallback"""
        
        # Create data directory
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.logger.info(f"Saving data to SQLite: {db_path}")
        
        conn = sqlite3.connect(db_path)
        
        try:
            # Save telematics data
            df.to_sql('telematics_data', conn, if_exists='replace', index=False)
            
            # Save driver profiles
            if self.driver_profiles:
                profiles_df = pd.DataFrame(self.driver_profiles)
                profiles_df.to_sql('driver_profiles', conn, if_exists='replace', index=False)
            
            # Save Austin incidents
            if self.incident_data:
                incidents_df = pd.DataFrame(self.incident_data)
                incidents_df.to_sql('austin_incidents', conn, if_exists='replace', index=False)
            
            
            # Save sample data for sharing
            sample_size = min(50, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            sample_df.to_csv('data/sample_telematics.csv', index=False)

            sample_df2 = profiles_df.sample(n=10, random_state=42)
            sample_df2.to_csv('data/sample_driver_profiles.csv', index=False)

            sample_df2 = incidents_df.sample(n=10, random_state=42)
            sample_df2.to_csv('data/sample_austin_incidents.csv', index=False)

            
            self.logger.info(f"SQLite data saved successfully")
            self.logger.info(f"Sample data: data/sample_telematics.csv")
            
        except Exception as e:
            self.logger.error(f"Error saving to SQLite: {e}")
        finally:
            conn.close()
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive dataset summary report"""
        
        summary = {
            'basic_stats': {
                'total_records': len(df),
                'unique_drivers': df['driver_id'].nunique(),
                'unique_trips': df['trip_id'].nunique(),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                }
            },
            'driver_behavior': {
                'avg_speed_kmh': round(df['speed_kmh'].mean(), 2),
                'max_speed_kmh': round(df['speed_kmh'].max(), 2),
                'hard_braking_rate': round(df['hard_braking'].mean() * 100, 2),
                'hard_acceleration_rate': round(df['hard_acceleration'].mean() * 100, 2),
                'phone_usage_rate': round(df['phone_usage'].mean() * 100, 2),
            },
            'temporal_patterns': {
                'rush_hour_percentage': round(df[(df['time_of_day'].isin([7, 8, 17, 18])) & 
                                                (~df['is_weekend'])].shape[0] / len(df) * 100, 2),
                'weekend_percentage': round(df['is_weekend'].mean() * 100, 2),
                'night_driving_percentage': round(df[df['time_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5])].shape[0] / len(df) * 100, 2)
            },
            'risk_assessment': {
                'avg_incident_risk': round(df['incident_risk_factor'].mean(), 3),
                'high_risk_events': df[df['incident_risk_factor'] > 2.0].shape[0],
                'avg_data_quality': round(df['data_quality_score'].mean(), 3)
            },
            'weather_distribution': df['weather_condition'].value_counts().to_dict(),
            'location_coverage': {
                'unique_start_locations': df['start_location'].nunique(),
                'unique_end_locations': df['end_location'].nunique(),
                'most_common_routes': df.groupby(['start_location', 'end_location']).size().nlargest(5).to_dict()
            }
        }
        
        return summary

def main():
    """Enhanced main function with comprehensive data handling and reporting"""
    
    print("Austin Telematics Insurance POC - Enhanced Data Generator")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize generator
        generator = AustinTelematicsGenerator(use_snowflake=True)
        
        # Generate synthetic dataset
        print("Generating comprehensive dataset...")
        df = generator.generate_synthetic_dataset(num_drivers=100, days=60)
        
        # Generate summary report
        summary = generator.generate_summary_report(df)
        
        # Try to save to Snowflake first
        snowflake_success = generator.save_to_snowflake(df)
        
        if not snowflake_success:
            print("Snowflake upload failed, using SQLite fallback...")
            generator.save_to_database(df)
        
        # Display comprehensive summary
        print("\n" + "=" * 70)
        print("DATASET GENERATION SUMMARY")
        print("=" * 70)
        
        print(f"\nBasic Statistics:")
        print(f"   Total records: {summary['basic_stats']['total_records']:,}")
        print(f"   Unique drivers: {summary['basic_stats']['unique_drivers']}")
        print(f"   Unique trips: {summary['basic_stats']['unique_trips']:,}")
        print(f"   Days covered: {summary['basic_stats']['date_range']['days']}")
        
        print(f"\nDriver Behavior Analysis:")
        print(f"   Average speed: {summary['driver_behavior']['avg_speed_kmh']} km/h")
        print(f"   Maximum speed: {summary['driver_behavior']['max_speed_kmh']} km/h")
        print(f"   Hard braking rate: {summary['driver_behavior']['hard_braking_rate']}%")
        print(f"   Hard acceleration rate: {summary['driver_behavior']['hard_acceleration_rate']}%")
        print(f"   Phone usage rate: {summary['driver_behavior']['phone_usage_rate']}%")
        
        print(f"\nTemporal Patterns:")
        print(f"   Rush hour driving: {summary['temporal_patterns']['rush_hour_percentage']}%")
        print(f"   Weekend driving: {summary['temporal_patterns']['weekend_percentage']}%")
        print(f"   Night driving: {summary['temporal_patterns']['night_driving_percentage']}%")
        
        print(f"\nWeather Distribution:")
        for weather, count in summary['weather_distribution'].items():
            percentage = (count / summary['basic_stats']['total_records']) * 100
            print(f"   {weather.title()}: {percentage:.1f}%")
        
        print(f"\nRisk Assessment:")
        print(f"   Average incident risk: {summary['risk_assessment']['avg_incident_risk']}")
        print(f"   High-risk events: {summary['risk_assessment']['high_risk_events']:,}")
        print(f"   Average data quality: {summary['risk_assessment']['avg_data_quality']}")
        
        print(f"\nAustin Coverage:")
        print(f"   Austin incidents integrated: {len(generator.incident_data)}")
        print(f"   Unique start locations: {summary['location_coverage']['unique_start_locations']}")
        print(f"   Unique end locations: {summary['location_coverage']['unique_end_locations']}")
        
        print(f"\nData Storage:")
        if snowflake_success:
            print("Primary: Snowflake (TELEMATICS_DATA, DRIVER_PROFILES, AUSTIN_INCIDENTS)")
        else:
            print("Primary: SQLite (austin_telematics.db)")
        print("Sample: data/sample_telematics.csv")
        print("Logs: logs/telematics_generator.log")
        
        print(f"\nReady for Next Steps:")
        print("ML model training (risk scoring)")
        print("Dynamic pricing implementation")
        print("Streamlit dashboard development")
        print("Advanced analytics and insights")
        
        print("\n" + "=" * 70)
        print("DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Data generation failed: {e}")
        print("Please check logs/telematics_generator.log for detailed error information")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        sys.exit(1)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
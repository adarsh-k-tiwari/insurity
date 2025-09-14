"""
Dynamic Insurance Pricing Engine for Austin Telematics POC
Implements Usage-Based Insurance (UBI) models: PAYD and PHYD
Integrates with ML risk scoring for personalized premium calculations
"""

import pandas as pd
import numpy as np
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricingConfig:
    """Configuration parameters for insurance pricing"""
    
    # Base pricing parameters
    BASE_ANNUAL_PREMIUM = 1200.0  # Base premium in USD
    
    # Risk adjustment ranges
    MIN_RISK_MULTIPLIER = 0.65   # Maximum discount (35% off)
    MAX_RISK_MULTIPLIER = 1.80   # Maximum surcharge (80% increase)
    
    # Usage-based factors
    LOW_MILEAGE_THRESHOLD = 8000   # Annual miles
    HIGH_MILEAGE_THRESHOLD = 20000 # Annual miles
    
    # Behavioral thresholds for discounts/penalties
    SAFE_DRIVER_THRESHOLD = 0.40   # Risk score for safe driver discounts
    HIGH_RISK_THRESHOLD = 0.60     # Risk score for high-risk penalties
    
    # Telematics program incentives
    PARTICIPATION_DISCOUNT = 0.05  # 5% discount for participating
    SAFE_DRIVER_BONUS = 0.10      # Additional 10% for consistently safe drivers
    IMPROVEMENT_BONUS = 0.03       # 3% for improving risk scores
    
    # Demographic adjustments (traditional factors)
    AGE_ADJUSTMENTS = {
        (18, 25): 1.15,   # Young drivers
        (26, 35): 1.00,   # Base rate
        (36, 50): 0.95,   # Experienced drivers
        (51, 65): 0.90,   # Mature drivers
        (66, 100): 1.05   # Senior drivers
    }
    
    # Vehicle adjustments
    VEHICLE_YEAR_ADJUSTMENTS = {
        'new': 1.10,      # 2020-2024
        'recent': 1.00,   # 2015-2019  
        'older': 0.95     # Pre-2015
    }

class UsageBasedPricingEngine:
    """Core pricing engine implementing PAYD and PHYD models"""
    
    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()
        self.connection = None
        
    def connect_to_snowflake(self):
        """Establish Snowflake connection"""
        try:
            self.connection = snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'TELEMATICS_DB'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'RAW_DATA'),
                client_session_keep_alive=True
            )
            logger.info("Connected to Snowflake")
            return True
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            return False
    
    def load_driver_data(self) -> pd.DataFrame:
        """Load driver data with telematics metrics"""
        
        if not self.connection:
            if not self.connect_to_snowflake():
                raise Exception("Cannot connect to Snowflake")
        
        query = """
        WITH driver_metrics AS (
            SELECT 
                t.DRIVER_ID,
                p.DRIVER_TYPE,
                p.AGE,
                p.VEHICLE_MAKE,
                p.VEHICLE_YEAR,
                
                -- Usage metrics
                COUNT(DISTINCT t.TRIP_ID) as TOTAL_TRIPS,
                COUNT(*) as TOTAL_DATA_POINTS,
                SUM(t.TRIP_DISTANCE_KM) as total_distance_km,
                
                -- Behavioral metrics
                AVG(t.SPEED_KMH) as AVG_SPEED_KMH,
                MAX(t.SPEED_KMH) as MAX_SPEED_KMH,
                SUM(CASE WHEN t.HARD_BRAKING = TRUE THEN 1 ELSE 0 END) as HARD_BRAKING_EVENTS,
                SUM(CASE WHEN t.HARD_ACCELERATION = TRUE THEN 1 ELSE 0 END) as hard_accel_events,
                SUM(CASE WHEN t.PHONE_USAGE = TRUE THEN 1 ELSE 0 END) as PHONE_USAGE_EVENTS,
                
                -- Time-based metrics
                SUM(CASE WHEN t.TIME_OF_DAY >= 22 OR t.TIME_OF_DAY <= 5 THEN 1 ELSE 0 END) as NIGHT_DRIVING_POINTS,
                SUM(CASE WHEN t.TIME_OF_DAY IN (7,8,17,18) AND t.IS_WEEKEND = FALSE THEN 1 ELSE 0 END) as rush_hour_points,
                
                -- Risk metrics
                AVG(t.INCIDENT_RISK_FACTOR) as avg_incident_risk,
                AVG(t.DATA_QUALITY_SCORE) as AVG_DATA_QUALITY,
                
                -- Date range for mileage calculation
                MIN(TRY_TO_TIMESTAMP(t.TIMESTAMP)) as first_trip_date,
                MAX(TRY_TO_TIMESTAMP(t.TIMESTAMP)) as last_trip_date,
                DATEDIFF('day', MIN(TRY_TO_TIMESTAMP(t.TIMESTAMP)), MAX(TRY_TO_TIMESTAMP(t.TIMESTAMP))) as days_of_data
                
            FROM TELEMATICS_DATA t
            LEFT JOIN DRIVER_PROFILES p ON t.DRIVER_ID = p.DRIVER_ID
            WHERE t.DATA_QUALITY_SCORE >= 0.7
            GROUP BY t.DRIVER_ID, p.DRIVER_TYPE, p.AGE, p.VEHICLE_MAKE, p.VEHICLE_YEAR
        )
        SELECT 
            DRIVER_ID,
            DRIVER_TYPE,
            AGE,
            VEHICLE_MAKE,
            VEHICLE_YEAR,
            TOTAL_TRIPS,
            TOTAL_DATA_POINTS,
            total_distance_km,
            AVG_SPEED_KMH,
            MAX_SPEED_KMH,
            HARD_BRAKING_EVENTS,
            hard_accel_events,
            PHONE_USAGE_EVENTS,
            NIGHT_DRIVING_POINTS,
            rush_hour_points,
            avg_incident_risk,
            AVG_DATA_QUALITY,
            first_trip_date,
            last_trip_date,
            days_of_data,
            -- Calculate annualized metrics with null safety
            CASE 
                WHEN days_of_data > 0 THEN (total_distance_km * 365.25 / days_of_data)
                ELSE total_distance_km * 365.25  -- Fallback if days calculation fails
            END as ANNUAL_MILEAGE_ESTIMATE,
            (TOTAL_DATA_POINTS * 0.5 / 60.0) as TOTAL_DRIVING_HOURS  -- 30-sec intervals to hours
        FROM driver_metrics
        WHERE days_of_data IS NOT NULL
        """
        
        logger.info("Loading driver data for pricing calculations...")
        df = pd.read_sql(query, self.connection)
        logger.info(f"Loaded data for {len(df)} drivers")
        
        return df
    
    def load_risk_scores(self) -> Dict[str, float]:
        """Load ML-generated risk scores from Snowflake RISK_SCORES table"""
        
        if not self.connection:
            if not self.connect_to_snowflake():
                logger.warning("Cannot connect to Snowflake, trying local file")
                return self.load_risk_scores_from_file()
        
        try:
            # Try loading from Snowflake first
            query = """
            SELECT DRIVER_ID, COMPOSITE_RISK_SCORE
            FROM RISK_SCORES
            """
            
            risk_df = pd.read_sql(query, self.connection)
            risk_scores = dict(zip(risk_df['DRIVER_ID'], risk_df['COMPOSITE_RISK_SCORE']))
            logger.info(f"Loaded risk scores from Snowflake for {len(risk_scores)} drivers")
            return risk_scores
            
        except Exception as e:
            logger.warning(f"Failed to load from Snowflake: {e}, trying local file")
            return self.load_risk_scores_from_file()
    
    def load_risk_scores_from_file(self) -> Dict[str, float]:
        """Load risk scores from local CSV file as fallback"""
        try:
            risk_df = pd.read_csv('data/driver_risk_scores.csv')
            risk_scores = dict(zip(risk_df['DRIVER_ID'], risk_df['composite_risk_score']))
            logger.info(f"Loaded risk scores from local file for {len(risk_scores)} drivers")
            return risk_scores
            
        except FileNotFoundError:
            logger.warning("Risk scores file not found, using default scores")
            return {}
    
    def calculate_payd_premium(self, driver_data: pd.Series) -> Dict[str, float]:
        """
        Calculate Pay-As-You-Drive (PAYD) premium based on mileage
        """
        
        annual_mileage = driver_data['ANNUAL_MILEAGE_ESTIMATE']
        base_premium = self.config.BASE_ANNUAL_PREMIUM
        
        # Mileage-based adjustments
        if annual_mileage < self.config.LOW_MILEAGE_THRESHOLD:
            mileage_multiplier = 0.85  # 15% discount for low mileage
        elif annual_mileage > self.config.HIGH_MILEAGE_THRESHOLD:
            mileage_multiplier = 1.20  # 20% surcharge for high mileage
        else:
            # Linear scaling between thresholds
            mileage_range = self.config.HIGH_MILEAGE_THRESHOLD - self.config.LOW_MILEAGE_THRESHOLD
            position = (annual_mileage - self.config.LOW_MILEAGE_THRESHOLD) / mileage_range
            mileage_multiplier = 0.85 + (position * 0.35)  # Scale from 0.85 to 1.20
        
        payd_premium = base_premium * mileage_multiplier
        
        return {
            'payd_base_premium': base_premium,
            'mileage_multiplier': round(mileage_multiplier, 3),
            'payd_premium': round(payd_premium, 2),
            'annual_mileage': round(annual_mileage, 0)
        }
    
    def calculate_phyd_premium(self, driver_data: pd.Series, risk_score: float) -> Dict[str, float]:
        """
        Calculate Pay-How-You-Drive (PHYD) premium based on behavior
        """
        
        base_premium = self.config.BASE_ANNUAL_PREMIUM
        
        # Risk-based multiplier (primary factor)
        risk_multiplier = (
            self.config.MIN_RISK_MULTIPLIER + 
            risk_score * (self.config.MAX_RISK_MULTIPLIER - self.config.MIN_RISK_MULTIPLIER)
        )
        
        # Additional behavioral adjustments
        behavioral_adjustments = 1.0
        
        # Hard braking penalty
        hard_braking_rate = driver_data['HARD_BRAKING_EVENTS'] / max(driver_data['TOTAL_DATA_POINTS'], 1)
        if hard_braking_rate > 0.15:  # More than 15% of data points
            behavioral_adjustments *= 1.10
        elif hard_braking_rate < 0.05:  # Less than 5%
            behavioral_adjustments *= 0.95
        
        # Phone usage penalty
        phone_usage_rate = driver_data['PHONE_USAGE_EVENTS'] / max(driver_data['TOTAL_DATA_POINTS'], 1)
        if phone_usage_rate > 0.10:
            behavioral_adjustments *= 1.15
        elif phone_usage_rate < 0.02:
            behavioral_adjustments *= 0.98
        
        # Night driving adjustment
        night_driving_rate = driver_data['NIGHT_DRIVING_POINTS'] / max(driver_data['TOTAL_DATA_POINTS'], 1)
        if night_driving_rate > 0.20:
            behavioral_adjustments *= 1.05
        
        # Speed behavior adjustment
        if driver_data['MAX_SPEED_KMH'] > 120:
            behavioral_adjustments *= 1.08
        elif driver_data['AVG_SPEED_KMH'] < 40:  # Very conservative driving
            behavioral_adjustments *= 0.97
        
        # Apply all adjustments
        final_multiplier = risk_multiplier * behavioral_adjustments
        phyd_premium = base_premium * final_multiplier
        
        return {
            'phyd_base_premium': base_premium,
            'risk_multiplier': round(risk_multiplier, 3),
            'behavioral_adjustments': round(behavioral_adjustments, 3),
            'final_multiplier': round(final_multiplier, 3),
            'phyd_premium': round(phyd_premium, 2)
        }
    
    def calculate_demographic_adjustments(self, driver_data: pd.Series) -> Dict[str, float]:
        """Calculate traditional demographic-based adjustments"""
        
        age = driver_data['AGE']
        vehicle_year = driver_data['VEHICLE_YEAR']
        
        # Age adjustment
        age_multiplier = 1.0
        for age_range, multiplier in self.config.AGE_ADJUSTMENTS.items():
            if age_range[0] <= age <= age_range[1]:
                age_multiplier = multiplier
                break
        
        # Vehicle year adjustment
        current_year = datetime.now().year
        if vehicle_year >= current_year - 4:
            vehicle_multiplier = self.config.VEHICLE_YEAR_ADJUSTMENTS['new']
        elif vehicle_year >= current_year - 9:
            vehicle_multiplier = self.config.VEHICLE_YEAR_ADJUSTMENTS['recent']
        else:
            vehicle_multiplier = self.config.VEHICLE_YEAR_ADJUSTMENTS['older']
        
        return {
            'age_multiplier': age_multiplier,
            'vehicle_multiplier': vehicle_multiplier,
            'demographic_multiplier': round(age_multiplier * vehicle_multiplier, 3)
        }
    
    def calculate_telematics_incentives(self, driver_data: pd.Series, risk_score: float) -> Dict[str, float]:
        """Calculate telematics program incentives and bonuses"""
        
        incentives = {
            'participation_discount': self.config.PARTICIPATION_DISCOUNT,
            'safe_driver_bonus': 0.0,
            'improvement_bonus': 0.0,
            'total_telematics_discount': 0.0
        }
        
        # Safe driver bonus
        if risk_score < self.config.SAFE_DRIVER_THRESHOLD:
            incentives['safe_driver_bonus'] = self.config.SAFE_DRIVER_BONUS
        
        # Data quality bonus (good telematics participation)
        if driver_data['AVG_DATA_QUALITY'] > 0.9 and driver_data['TOTAL_TRIPS'] > 50:
            incentives['improvement_bonus'] = self.config.IMPROVEMENT_BONUS
        
        # Calculate total discount
        incentives['total_telematics_discount'] = (
            incentives['participation_discount'] + 
            incentives['safe_driver_bonus'] + 
            incentives['improvement_bonus']
        )
        
        return incentives
    
    def calculate_comprehensive_premium(self, driver_data: pd.Series, risk_score: float) -> Dict[str, any]:
        """Calculate comprehensive premium using all pricing models"""
        
        # Calculate individual pricing models
        payd_results = self.calculate_payd_premium(driver_data)
        phyd_results = self.calculate_phyd_premium(driver_data, risk_score)
        demographic_adj = self.calculate_demographic_adjustments(driver_data)
        incentives = self.calculate_telematics_incentives(driver_data, risk_score)
        
        # Hybrid model: Combine PAYD and PHYD
        payd_weight = 0.3  # 30% usage-based
        phyd_weight = 0.7  # 70% behavior-based
        
        hybrid_premium = (
            payd_results['payd_premium'] * payd_weight +
            phyd_results['phyd_premium'] * phyd_weight
        )
        
        # Apply demographic adjustments
        demographically_adjusted = hybrid_premium * demographic_adj['demographic_multiplier']
        
        # Apply telematics incentives
        telematics_discount_amount = demographically_adjusted * incentives['total_telematics_discount']
        final_premium = demographically_adjusted - telematics_discount_amount
        
        # Calculate traditional premium for comparison
        traditional_premium = (
            self.config.BASE_ANNUAL_PREMIUM * 
            demographic_adj['demographic_multiplier']
        )
        
        # Calculate savings
        premium_savings = traditional_premium - final_premium
        savings_percentage = (premium_savings / traditional_premium) * 100 if traditional_premium > 0 else 0
        
        return {
            'driver_id': driver_data['DRIVER_ID'],
            'driver_type': driver_data['DRIVER_TYPE'],
            'age': driver_data['AGE'],
            'risk_score': risk_score,
            
            # Individual model results
            'payd_premium': payd_results['payd_premium'],
            'phyd_premium': phyd_results['phyd_premium'],
            'traditional_premium': round(traditional_premium, 2),
            
            # Hybrid model results
            'hybrid_base_premium': round(hybrid_premium, 2),
            'demographic_adjusted_premium': round(demographically_adjusted, 2),
            'telematics_discount_amount': round(telematics_discount_amount, 2),
            'final_telematics_premium': round(final_premium, 2),
            
            # Savings analysis
            'premium_savings': round(premium_savings, 2),
            'savings_percentage': round(savings_percentage, 1),
            
            # Model components for transparency
            'mileage_multiplier': payd_results['mileage_multiplier'],
            'risk_multiplier': phyd_results['final_multiplier'],
            'demographic_multiplier': demographic_adj['demographic_multiplier'],
            'total_telematics_discount': round(incentives['total_telematics_discount'], 3),
            
            # Usage metrics
            'annual_mileage': payd_results['annual_mileage'],
            'TOTAL_TRIPS': driver_data['TOTAL_TRIPS'],
            'TOTAL_DRIVING_HOURS': round(driver_data['TOTAL_DRIVING_HOURS'], 1),
            
            'calculation_date': datetime.now().isoformat()
        }
    
    def save_pricing_to_snowflake(self, results_df: pd.DataFrame) -> bool:
        """Save pricing results to Snowflake PRICING_DATA table"""
        
        if not self.connection:
            if not self.connect_to_snowflake():
                logger.error("Cannot connect to Snowflake for saving pricing data")
                return False
        
        try:
            logger.info("Uploading pricing data to Snowflake...")
            
            # Prepare data for Snowflake table structure
            policy_period = f"2025-{datetime.now().strftime('%m')}-01_2026-{datetime.now().strftime('%m')}-01"
            
            pricing_upload = pd.DataFrame({
                'DRIVER_ID': results_df['driver_id'],
                'POLICY_PERIOD': policy_period,
                'BASE_PREMIUM': results_df['traditional_premium'],
                'RISK_ADJUSTMENT': results_df['risk_multiplier'],
                'FINAL_PREMIUM': results_df['final_telematics_premium'],
                'DISCOUNT_APPLIED': results_df['total_telematics_discount']
                # Note: CREATED_AT will be set by Snowflake DEFAULT CURRENT_TIMESTAMP()
            })
            
            # Upload to Snowflake
            success, _, nrows, _ = write_pandas(
                self.connection, 
                pricing_upload, 
                'PRICING_DATA', 
                auto_create_table=False, 
                overwrite=True
            )
            
            if success:
                logger.info(f"Successfully uploaded {nrows:,} pricing records to Snowflake")
                return True
            else:
                logger.error("Failed to upload pricing data to Snowflake")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading pricing data to Snowflake: {e}")
            return False
    
    def process_all_drivers(self) -> pd.DataFrame:
        """Process premium calculations for all drivers"""
        
        logger.info("Starting comprehensive premium calculations...")
        
        # Load data
        driver_data = self.load_driver_data()
        risk_scores = self.load_risk_scores()
        
        # Calculate premiums for each driver
        results = []
        for _, driver_row in driver_data.iterrows():
            driver_id = driver_row['DRIVER_ID']
            risk_score = risk_scores.get(driver_id, 0.5)  # Default to medium risk
            
            premium_calc = self.calculate_comprehensive_premium(driver_row, risk_score)
            results.append(premium_calc)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Completed premium calculations for {len(results_df)} drivers")
        
        return results_df
    
    def generate_pricing_summary(self, results_df: pd.DataFrame) -> Dict[str, any]:
        """Generate summary statistics for pricing analysis"""
        
        summary = {
            'total_drivers': len(results_df),
            'average_traditional_premium': round(results_df['traditional_premium'].mean(), 2),
            'average_telematics_premium': round(results_df['final_telematics_premium'].mean(), 2),
            'total_potential_savings': round(results_df['premium_savings'].sum(), 2),
            'average_savings_percentage': round(results_df['savings_percentage'].mean(), 1),
            
            'premium_ranges': {
                'traditional': {
                    'min': round(results_df['traditional_premium'].min(), 2),
                    'max': round(results_df['traditional_premium'].max(), 2),
                    'range': round(results_df['traditional_premium'].max() - results_df['traditional_premium'].min(), 2)
                },
                'telematics': {
                    'min': round(results_df['final_telematics_premium'].min(), 2),
                    'max': round(results_df['final_telematics_premium'].max(), 2),
                    'range': round(results_df['final_telematics_premium'].max() - results_df['final_telematics_premium'].min(), 2)
                }
            },
            
            'risk_distribution': results_df['driver_type'].value_counts().to_dict(),
            
            'business_metrics': {
                'revenue_impact_percentage': round(
                    (results_df['final_telematics_premium'].sum() / results_df['traditional_premium'].sum() - 1) * 100, 2
                ),
                'drivers_saving_money': len(results_df[results_df['premium_savings'] > 0]),
                'drivers_paying_more': len(results_df[results_df['premium_savings'] < 0]),
                'max_discount_offered': round(results_df['savings_percentage'].max(), 1),
                'max_surcharge_applied': round(results_df['savings_percentage'].min(), 1)
            }
        }
        
        return summary
    
    def save_results(self, results_df: pd.DataFrame, filename: str = 'data/pricing_results.csv'):
        """Save pricing results to file"""
        
        os.makedirs('data', exist_ok=True)
        results_df.to_csv(filename, index=False)
        logger.info(f"Pricing results saved to {filename}")
    
    def close_connection(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")

def main():
    """Main execution function"""
    
    print("Dynamic Insurance Pricing Engine - Austin Telematics POC")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize pricing engine
        pricing_engine = UsageBasedPricingEngine()
        
        # Process all drivers
        results_df = pricing_engine.process_all_drivers()
        
        # Generate summary
        summary = pricing_engine.generate_pricing_summary(results_df)
        
        # Save results locally
        pricing_engine.save_results(results_df)
        
        # Save to Snowflake
        snowflake_success = pricing_engine.save_pricing_to_snowflake(results_df)
        
        # Display results
        print("=" * 70)
        print("DYNAMIC PRICING RESULTS")
        print("=" * 70)
        
        print(f"\nPortfolio Summary:")
        print(f"   Total drivers: {summary['total_drivers']}")
        print(f"   Average traditional premium: ${summary['average_traditional_premium']:,.2f}")
        print(f"   Average telematics premium: ${summary['average_telematics_premium']:,.2f}")
        print(f"   Average savings: {summary['average_savings_percentage']}%")
        
        print(f"\nBusiness Impact:")
        print(f"   Total potential savings: ${summary['total_potential_savings']:,.2f}")
        print(f"   Revenue impact: {summary['business_metrics']['revenue_impact_percentage']}%")
        print(f"   Drivers saving money: {summary['business_metrics']['drivers_saving_money']}")
        print(f"   Drivers paying more: {summary['business_metrics']['drivers_paying_more']}")
        
        print(f"\nPremium Differentiation:")
        print(f"   Traditional range: ${summary['premium_ranges']['traditional']['min']:,.2f} - ${summary['premium_ranges']['traditional']['max']:,.2f}")
        print(f"   Telematics range: ${summary['premium_ranges']['telematics']['min']:,.2f} - ${summary['premium_ranges']['telematics']['max']:,.2f}")
        print(f"   Max discount: {summary['business_metrics']['max_discount_offered']}%")
        print(f"   Max surcharge: {abs(summary['business_metrics']['max_surcharge_applied'])}%")
        
        print(f"\nTop 5 Premium Winners & Losers:")
        winners = results_df.nlargest(5, 'savings_percentage')[['driver_id', 'driver_type', 'savings_percentage', 'premium_savings']]
        losers = results_df.nsmallest(5, 'savings_percentage')[['driver_id', 'driver_type', 'savings_percentage', 'premium_savings']]
        
        print("   Biggest Savers:")
        for _, driver in winners.iterrows():
            print(f"     {driver['driver_id']} ({driver['driver_type']}): {driver['savings_percentage']}% (${abs(driver['premium_savings']):.2f} saved)")
        
        print("   Highest Surcharges:")
        for _, driver in losers.iterrows():
            print(f"     {driver['driver_id']} ({driver['driver_type']}): {abs(driver['savings_percentage'])}% (${abs(driver['premium_savings']):.2f} increase)")
        
        print(f"\nData Storage:")
        if snowflake_success:
            print("   Pricing data successfully uploaded to Snowflake PRICING_DATA table")
        else:
            print("   Snowflake upload failed - check data/pricing_results.csv")
        print("   Detailed results: data/pricing_results.csv")
        
        print(f"\nReady for Next Steps:")
        print("   Streamlit dashboard integration")
        print("   Business case presentation")
        
        print("\n" + "=" * 70)
        print("DYNAMIC PRICING ENGINE COMPLETED!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        return False
    finally:
        pricing_engine.close_connection()

if __name__ == "__main__":
    success = main()
    
    if not success:
        exit(1)
        
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Dynamic pricing engine ready for insurance deployment!")
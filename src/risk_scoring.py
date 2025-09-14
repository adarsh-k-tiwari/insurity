"""
Advanced ML Risk Scoring System for Austin Telematics Insurance POC
Implements ensemble methods (Random Forest + XGBoost + Neural Network) for driver risk assessment
"""

import pandas as pd
import numpy as np
import snowflake.connector
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SnowflakeDataLoader:
    """Handle Snowflake data loading and preprocessing"""
    
    def __init__(self):
        self.connection = None
        self.scaler = StandardScaler()
        
    def connect_to_snowflake(self):
        """Establish Snowflake connection"""
        try:
            self.connection = snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'TELEMATICS_DB'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'RAW_DATA')
            )
            logger.info("Connected to Snowflake")
            return True
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            return False
    
    def load_telematics_data(self) -> pd.DataFrame:
        """Load and preprocess telematics data from Snowflake"""
        
        if not self.connection:
            if not self.connect_to_snowflake():
                raise Exception("Cannot connect to Snowflake")
        
        query = """
        SELECT 
            t.*,
            p.DRIVER_TYPE,
            p.AGE,
            p.VEHICLE_MAKE,
            p.VEHICLE_YEAR,
            p.BASE_SPEED_MULTIPLIER,
            p.HARD_BRAKING_PROB,
            p.HARD_ACCEL_PROB
        FROM TELEMATICS_DATA t
        LEFT JOIN DRIVER_PROFILES p ON t.DRIVER_ID = p.DRIVER_ID
        WHERE t.DATA_QUALITY_SCORE >= 0.7
        """
        
        logger.info("Loading telematics data from Snowflake...")
        df = pd.read_sql(query, self.connection)
        logger.info(f"Loaded {len(df):,} records from Snowflake")
        
        return df
    
    def close_connection(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")

class FeatureEngineer:
    """Advanced feature engineering for telematics data"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced behavioral features from telematics data"""
        
        logger.info("Engineering behavioral features...")
        
        # Create a copy to avoid modifying original data
        df_features = df.copy()
        
        # Time-based features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['TIME_OF_DAY'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['TIME_OF_DAY'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['DAY_OF_WEEK'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['DAY_OF_WEEK'] / 7)
        
        # Speed-related features
        df_features['speed_variability'] = df_features.groupby('TRIP_ID')['SPEED_KMH'].transform('std').fillna(0)
        df_features['max_speed_in_trip'] = df_features.groupby('TRIP_ID')['SPEED_KMH'].transform('max')
        df_features['speed_percentile_90'] = df_features.groupby('TRIP_ID')['SPEED_KMH'].transform(lambda x: x.quantile(0.9))
        
        # Acceleration features
        df_features['acceleration_variability'] = df_features.groupby('TRIP_ID')['ACCELERATION_MS2'].transform('std').fillna(0)
        df_features['extreme_acceleration_count'] = df_features.groupby('TRIP_ID')['ACCELERATION_MS2'].transform(
            lambda x: (x.abs() > 3.0).sum()
        )
        
        # Trip-level aggregations
        trip_stats = df_features.groupby('TRIP_ID').agg({
            'HARD_BRAKING': 'sum',
            'HARD_ACCELERATION': 'sum', 
            'PHONE_USAGE': 'sum',
            'SPEED_KMH': ['mean', 'std', 'max'],
            'ACCELERATION_MS2': ['std', 'min', 'max'],
            'INCIDENT_RISK_FACTOR': 'mean'
        }).round(3)
        
        # Flatten column names
        trip_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in trip_stats.columns]
        trip_stats = trip_stats.add_prefix('trip_')
        
        # Merge back to main dataframe
        df_features = df_features.merge(trip_stats, left_on='TRIP_ID', right_index=True, how='left')
        
        # Driver-level historical features
        driver_stats = df_features.groupby('DRIVER_ID').agg({
            'HARD_BRAKING': 'mean',
            'HARD_ACCELERATION': 'mean',
            'PHONE_USAGE': 'mean',
            'SPEED_KMH': 'mean',
            'INCIDENT_RISK_FACTOR': 'mean'
        }).round(3)
        
        driver_stats = driver_stats.add_prefix('driver_avg_')
        df_features = df_features.merge(driver_stats, left_on='DRIVER_ID', right_index=True, how='left')
        
        # Weather impact features
        weather_encoder = LabelEncoder()
        df_features['weather_encoded'] = weather_encoder.fit_transform(df_features['WEATHER_CONDITION'].fillna('clear'))
        self.label_encoders['weather'] = weather_encoder
        
        # Location features
        location_encoder = LabelEncoder()
        df_features['start_location_encoded'] = location_encoder.fit_transform(df_features['START_LOCATION'].fillna('unknown'))
        self.label_encoders['location'] = location_encoder
        
        logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} new features")
        
        return df_features
    
    def create_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk score labels based on driving behavior"""
        
        logger.info("Creating risk score labels...")
        
        df_risk = df.copy()
        
        # Component scores (0-1 scale, higher = more risk)
        
        # 1. Speed Score (20% weight)
        speed_violations = ((df_risk['SPEED_KMH'] > 70).astype(int) * 0.8 + (df_risk['SPEED_KMH'] > 90).astype(int) * 0.4) 
        speed_variability_norm = (df_risk['speed_variability'] / df_risk['speed_variability'].quantile(0.95)).clip(0, 1)
        df_risk['speed_score'] = (0.6 * speed_violations + 0.4 * speed_variability_norm).clip(0, 1)
        
        # 2. Braking Score (20% weight) 
        df_risk['braking_score'] = df_risk['HARD_BRAKING'].astype(float).clip(0, 1)
        
        # 3. Acceleration Score (15% weight)
        df_risk['acceleration_score'] = df_risk['HARD_ACCELERATION'].astype(float).clip(0, 1)
        
        # 4. Phone Usage Score (30% weight)
        df_risk['phone_usage_score'] = df_risk['PHONE_USAGE'].astype(float).clip(0, 1)
        
        # 5. Time Score (10% weight) - higher risk for night/rush hour
        night_risk = ((df_risk['TIME_OF_DAY'] >= 22) | (df_risk['TIME_OF_DAY'] <= 5)).astype(float) * 1.2
        rush_hour_risk = (
            ((df_risk['TIME_OF_DAY'].isin([7, 8, 17, 18])) & (~df_risk['IS_WEEKEND']))
        ).astype(float) * 1.0
        df_risk['time_score'] = (night_risk + rush_hour_risk).clip(0, 1)
        
        # 6. Distance Score (10% weight) - based on distance from home
        distance_norm = (df_risk['DISTANCE_FROM_HOME'] / df_risk['DISTANCE_FROM_HOME'].quantile(0.95)).clip(0, 1)
        df_risk['distance_score'] = distance_norm * 0.5  # Lower weight for distance
        
        # 7. Incident Exposure Score (10% weight)
        incident_norm = ((df_risk['INCIDENT_RISK_FACTOR'] - 1) / 2).clip(0, 1)  # Normalize 1-3 scale to 0-1
        df_risk['incident_exposure_score'] = incident_norm
        
        # 8. Behavioral 
        df_risk['behavioral_amplifier'] = (
            (df_risk['HARD_BRAKING'] & df_risk['PHONE_USAGE']).astype(float) * 0.3 +  # Multiple bad behaviors
            (df_risk['speed_variability'] > df_risk['speed_variability'].quantile(0.8)).astype(float) * 0.2
        )

        # Overall risk score (weighted combination)
        weights = {
            'speed_score': 0.20,
            'braking_score': 0.20,
            'acceleration_score': 0.15,
            'phone_usage_score': 0.30,
            'time_score': 0.10,
            'distance_score': 0.10,
            'incident_exposure_score': 0.10
        }
        
        # Apply amplifier to overall score
        df_risk['overall_risk_score'] = (
            sum(df_risk[score] * weight for score, weight in weights.items()) + 
            df_risk['behavioral_amplifier']
        ).clip(0, 1)
        
        # Risk categories
        df_risk['risk_category'] = pd.cut(
            df_risk['overall_risk_score'],
            bins=[0, 0.2, 0.55, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        logger.info("Risk labels created successfully")
        logger.info(f"   Risk distribution: {df_risk['risk_category'].value_counts().to_dict()}")
        
        return df_risk

class EnsembleRiskModel:
    """Ensemble ML model for risk scoring using Random Forest + XGBoost + Neural Network"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for ML models"""
        
        # Select relevant features for modeling
        feature_columns = [
            # Basic telematics
            'SPEED_KMH', 'ACCELERATION_MS2', 'HARD_BRAKING', 'HARD_ACCELERATION', 'PHONE_USAGE',
            'TIME_OF_DAY', 'DAY_OF_WEEK', 'IS_WEEKEND', 'INCIDENT_RISK_FACTOR', 'DISTANCE_FROM_HOME',
            
            # Driver characteristics  
            'AGE', 'BASE_SPEED_MULTIPLIER', 'HARD_BRAKING_PROB', 'HARD_ACCEL_PROB',
            
            # Engineered features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'speed_variability', 'max_speed_in_trip', 'speed_percentile_90',
            'acceleration_variability', 'extreme_acceleration_count',
            'weather_encoded', 'start_location_encoded',
            
            # Trip-level features (if available)
            'trip_HARD_BRAKING_sum', 'trip_HARD_ACCELERATION_sum', 'trip_PHONE_USAGE_sum',
            'trip_SPEED_KMH_mean', 'trip_SPEED_KMH_std', 'trip_SPEED_KMH_max',
            
            # Driver historical averages
            'driver_avg_HARD_BRAKING', 'driver_avg_HARD_ACCELERATION', 'driver_avg_PHONE_USAGE'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        # Prepare feature matrix
        X = df[available_features].fillna(0).values
        y = df['overall_risk_score'].values
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        
        return X, y, available_features
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ensemble of ML models"""
        
        logger.info("Training ensemble ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        results = {}
        
        # 1. Random Forest Model
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'mse': mean_squared_error(y_test, rf_pred),
            'feature_importance': dict(zip(self.feature_names, rf_model.feature_importances_))
        }
        
        # 2. XGBoost Model
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        self.models['xgboost'] = xgb_model
        results['xgboost'] = {
            'mse': mean_squared_error(y_test, xgb_pred),
            'feature_importance': dict(zip(self.feature_names, xgb_model.feature_importances_))
        }
        
        # 3. Neural Network Model
        logger.info("Training Neural Network...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        
        self.models['neural_network'] = nn_model
        results['neural_network'] = {
            'mse': mean_squared_error(y_test, nn_pred),
            'feature_importance': {}
        }
        
        # 4. Ensemble Prediction (weighted average)
        ensemble_weights = {'random_forest': 0.4, 'xgboost': 0.4, 'neural_network': 0.2}
        ensemble_pred = (
            ensemble_weights['random_forest'] * rf_pred +
            ensemble_weights['xgboost'] * xgb_pred +
            ensemble_weights['neural_network'] * nn_pred
        )
        
        results['ensemble'] = {
            'mse': mean_squared_error(y_test, ensemble_pred),
            'weights': ensemble_weights
        }
        
        # Store test data for evaluation
        self.test_data = {
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'predictions': {
                'random_forest': rf_pred,
                'xgboost': xgb_pred,
                'neural_network': nn_pred,
                'ensemble': ensemble_pred
            }
        }
        
        self.is_trained = True
        logger.info("All models trained successfully")
        
        return results
    
    def predict_risk_score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate risk predictions using ensemble models"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {
            'random_forest': self.models['random_forest'].predict(X),
            'xgboost': self.models['xgboost'].predict(X),
            'neural_network': self.models['neural_network'].predict(X_scaled)
        }
        
        # Ensemble prediction
        ensemble_weights = {'random_forest': 0.4, 'xgboost': 0.4, 'neural_network': 0.2}
        predictions['ensemble'] = (
            ensemble_weights['random_forest'] * predictions['random_forest'] +
            ensemble_weights['xgboost'] * predictions['xgboost'] +
            ensemble_weights['neural_network'] * predictions['neural_network']
        )
        
        return predictions
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        evaluation = {}
        y_test = self.test_data['y_test']
        
        for model_name, predictions in self.test_data['predictions'].items():
            # Basic metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - predictions))
            r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            # Business metrics
            risk_categories_true = pd.cut(y_test, bins=[0, 0.3, 0.6, 1.0], labels=['LOW', 'MEDIUM', 'HIGH'])
            risk_categories_pred = pd.cut(predictions, bins=[0, 0.3, 0.6, 1.0], labels=['LOW', 'MEDIUM', 'HIGH'])
            category_accuracy = (risk_categories_true == risk_categories_pred).mean()
            
            evaluation[model_name] = {
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'r2_score': round(r2, 4),
                'category_accuracy': round(category_accuracy, 4)
            }
        
        logger.info("Model evaluation completed")
        
        return evaluation
    
    def save_models(self, model_dir: str = 'models/'):
        """Save trained models and scalers"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/risk_model_{name}.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{model_dir}/scaler_{name}.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, f'{model_dir}/feature_names.pkl')
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = 'models/'):
        """Load trained models and scalers"""
        
        # Load models
        model_files = {
            'random_forest': f'{model_dir}/risk_model_random_forest.pkl',
            'xgboost': f'{model_dir}/risk_model_xgboost.pkl', 
            'neural_network': f'{model_dir}/risk_model_neural_network.pkl'
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
        
        # Load scalers
        scaler_files = {'standard': f'{model_dir}/scaler_standard.pkl'}
        for name, filepath in scaler_files.items():
            if os.path.exists(filepath):
                self.scalers[name] = joblib.load(filepath)
        
        # Load feature names
        feature_file = f'{model_dir}/feature_names.pkl'
        if os.path.exists(feature_file):
            self.feature_names = joblib.load(feature_file)
        
        self.is_trained = len(self.models) > 0
        logger.info(f"Models loaded from {model_dir}")

class RiskScoringPipeline:
    """Complete pipeline for ML-based risk scoring"""
    
    def __init__(self):
        self.data_loader = SnowflakeDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model = EnsembleRiskModel()
        self.processed_data = None
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ML risk scoring pipeline"""
        
        logger.info("Starting ML Risk Scoring Pipeline...")
        
        try:
            # 1. Load data from Snowflake
            raw_data = self.data_loader.load_telematics_data()
            
            # 2. Engineer features
            featured_data = self.feature_engineer.create_behavioral_features(raw_data)
            
            # 3. Create risk labels
            labeled_data = self.feature_engineer.create_risk_labels(featured_data)
            self.processed_data = labeled_data
            
            # 4. Prepare features for ML
            X, y, feature_names = self.model.prepare_features(labeled_data)
            
            # 5. Train ensemble models
            training_results = self.model.train_models(X, y)
            
            # 6. Evaluate models
            evaluation_results = self.model.evaluate_models()
            
            # 7. Save models
            self.model.save_models()
            
            # 8. Generate comprehensive report
            pipeline_results = {
                'data_summary': {
                    'total_records': len(labeled_data),
                    'unique_drivers': labeled_data['DRIVER_ID'].nunique(),
                    'date_range': f"{labeled_data['TIMESTAMP'].min()} to {labeled_data['TIMESTAMP'].max()}",
                    'risk_distribution': labeled_data['risk_category'].value_counts().to_dict()
                },
                'feature_engineering': {
                    'total_features': len(feature_names),
                    'feature_names': feature_names
                },
                'model_training': training_results,
                'model_evaluation': evaluation_results
            }
            
            logger.info("ML Risk Scoring Pipeline completed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.data_loader.close_connection()
    
    def generate_driver_risk_scores(self) -> pd.DataFrame:
        """Generate risk scores for all drivers"""
        
        if self.processed_data is None:
            raise ValueError("Pipeline must be run first")
        
        logger.info("Generating driver-level risk scores...")
        
        # Aggregate to driver level
        driver_scores = self.processed_data.groupby('DRIVER_ID').agg({
            'overall_risk_score': 'mean',
            'speed_score': 'mean',
            'braking_score': 'mean', 
            'acceleration_score': 'mean',
            'phone_usage_score': 'mean',
            'time_score': 'mean',
            'distance_score': 'mean',
            'incident_exposure_score': 'mean',
            'DRIVER_TYPE': 'first',
            'AGE': 'first',
            'VEHICLE_MAKE': 'first'
        }).round(3)
        
        # Add risk categories
        driver_scores['risk_category'] = pd.cut(
            driver_scores['overall_risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Add premium impact estimates
        driver_scores['premium_multiplier'] = (
            0.7 + driver_scores['overall_risk_score'] * 1.35
        ).round(2)  # Range: 0.65x to 2.0x
        
        logger.info(f"Generated risk scores for {len(driver_scores)} drivers")
        
        return driver_scores.reset_index()

def main():
    """Main execution function"""
    
    print("Advanced ML Risk Scoring System - Austin Telematics POC")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize and run pipeline
        pipeline = RiskScoringPipeline()
        results = pipeline.run_complete_pipeline()
        
        # Generate driver risk scores
        driver_scores = pipeline.generate_driver_risk_scores()
        
        # Display results
        print("=" * 70)
        print("ML RISK SCORING RESULTS")
        print("=" * 70)
        
        print(f"\nData Summary:")
        print(f"   Total records processed: {results['data_summary']['total_records']:,}")
        print(f"   Unique drivers analyzed: {results['data_summary']['unique_drivers']}")
        print(f"   Features engineered: {results['feature_engineering']['total_features']}")
        
        print(f"\n🎯 Risk Distribution:")
        for category, count in results['data_summary']['risk_distribution'].items():
            percentage = (count / results['data_summary']['total_records']) * 100
            print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nModel Performance:")
        for model_name, metrics in results['model_evaluation'].items():
            print(f"   {model_name.replace('_', ' ').title()}:")
            print(f"      RMSE: {metrics['rmse']}")
            print(f"      R2 Score: {metrics['r2_score']}")
            print(f"      Category Accuracy: {metrics['category_accuracy']:.1%}")
        
        print(f"\n👥 Driver Risk Scores (Top 10 Riskiest):")
        top_risk_drivers = driver_scores.nlargest(10, 'overall_risk_score')[
            ['DRIVER_ID', 'DRIVER_TYPE', 'overall_risk_score', 'risk_category', 'premium_multiplier']
        ]
        print(top_risk_drivers.to_string(index=False))
        
        # Save results
        driver_scores.to_csv('data/driver_risk_scores.csv', index=False)
        
        print(f"\nOutput Files:")
        print("   Models saved to: models/")
        print("   Driver scores: data/driver_risk_scores.csv")
        
        print(f"\nReady for Next Steps:")
        print("   Dynamic pricing engine integration")
        print("   Streamlit dashboard development") 
        print("   Business impact analysis")
        
        print("\n" + "=" * 70)
        print("ML RISK SCORING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        exit(1)
        
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Advanced ML models ready for insurance pricing!")
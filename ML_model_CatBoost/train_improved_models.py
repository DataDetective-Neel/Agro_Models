# train_improved_models.py - CatBoost Version
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class ImprovedCropTrainer:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        
    def load_regional_data(self, region_name):
        """Load and enhance regional data"""
        file_path = f"data/processed/{region_name.lower()}_data.csv"
        df = pd.read_csv(file_path)
        df = self.create_better_features(df)
        return df
    
    def create_better_features(self, df):
        """Create enhanced features for better prediction"""
        df['NP_ratio'] = df['N'] / (df['P'] + 1)
        df['NK_ratio'] = df['N'] / (df['K'] + 1)
        df['PK_ratio'] = df['P'] / (df['K'] + 1)
        df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100
        df['rainfall_moisture_balance'] = df['Rainfall'] / (df['Moisture'] + 1)
        df['soil_suitability'] = df['pH'] * df['Moisture'] / 10
        season_strength = {'Rabi': 1, 'Kharif': 2, 'Zaid': 3}
        df['season_strength'] = df['Season'].map(season_strength)
        return df
    
    def train_region_model(self, region_name):
        """Train improved model for specific region"""
        print(f"\n🚀 Training IMPROVED CatBoost model for {region_name}...")
        
        df = self.load_regional_data(region_name)
        print(f"   Data shape after feature engineering: {df.shape}")
        
        features = [
            'N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Moisture',
            'NP_ratio', 'NK_ratio', 'PK_ratio', 'temp_humidity_index', 
            'rainfall_moisture_balance', 'soil_suitability', 'season_strength',
            'SoilType', 'Irrigation'
        ]
        target = 'Crop'
        
        X = df[features].copy()
        y = df[target]
        
        # CatBoost handles categorical features natively!
        categorical_features = ['SoilType', 'Irrigation']
        cat_indices = [features.index(f) for f in categorical_features]
        
        # Encode target
        le_crop = LabelEncoder()
        y_encoded = le_crop.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train IMPROVED CatBoost model
        model = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function='MultiClass',
            cat_features=cat_indices,  # CatBoost's killer feature!
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ {region_name} IMPROVED CatBoost model trained!")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Features used: {len(features)}")
        print(f"   Crops: {list(le_crop.classes_)}")
        
        # Store components
        self.models[region_name] = model
        self.encoders[region_name] = {'crop': le_crop}
        
        return accuracy
    
    def save_models(self):
        """Save all improved models"""
        print(f"\n💾 Saving IMPROVED CatBoost models...")
        
        for region in self.models:
            model_data = {
                'model': self.models[region],
                'encoders': self.encoders[region],
                'feature_names': self.get_feature_names()
            }
            
            file_path = f"ML models/{region.lower()}_improved_model.pkl"
            joblib.dump(model_data, file_path)
            print(f"✅ Saved {region} improved model")
    
    def get_feature_names(self):
        return [
            'N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Moisture',
            'NP_ratio', 'NK_ratio', 'PK_ratio', 'temp_humidity_index', 
            'rainfall_moisture_balance', 'soil_suitability', 'season_strength',
            'SoilType', 'Irrigation'
        ]

def main():
    print("IMPROVED REGIONAL CROP MODEL TRAINING - CatBoost")
    print("=" * 60)
    
    trainer = ImprovedCropTrainer()
    regions = ['Punjab', 'Haryana', 'Rajasthan']
    results = {}
    
    for region in regions:
        accuracy = trainer.train_region_model(region)
        results[region] = accuracy
    
    trainer.save_models()
    
    print(f"\n🎉 IMPROVED TRAINING COMPLETED!")
    print(f"📊 Regional Model Performance:")
    for region, accuracy in results.items():
        print(f"{region}: {accuracy:.2%}")

if __name__ == "__main__":
    main()

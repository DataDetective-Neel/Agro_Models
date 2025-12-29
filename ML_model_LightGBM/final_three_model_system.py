# final_three_model_system.py - LightGBM Version
import joblib
import pandas as pd
import numpy as np
import os

class FinalThreeModelSystem:
    def __init__(self):
        print("🚀 INITIALIZING 3-MODEL CROP SYSTEM (LightGBM)")
        print("=" * 50)
        
        self.models_loaded = {'model1': False, 'model2': False, 'model3': False}
        
        # Load Model 1: General high-accuracy model
        try:
            self.model1 = joblib.load('ML models/crop_classifier_model.pkl')
            self.scaler1 = joblib.load('ML models/scaler_clf.pkl')
            self.encoder1 = joblib.load('ML models/label_encoder.pkl')
            self.models_loaded['model1'] = True
            print("✅ MODEL 1: High-Accuracy LightGBM - LOADED")
        except Exception as e:
            print(f"❌ MODEL 1: Not available - {e}")
        
        # Load Model 2: Regional models
        self.regional_models = {}
        regions = ['punjab', 'haryana', 'rajasthan']
        for region in regions:
            try:
                model_data = joblib.load(f'ML models/{region}_improved_model.pkl')
                self.regional_models[region] = model_data
                print(f"✅ MODEL 2: {region.title()} Regional LightGBM - LOADED")
            except:
                try:
                    model_data = joblib.load(f'ML models/{region}_model.pkl')
                    self.regional_models[region] = model_data
                    print(f"✅ MODEL 2: {region.title()} Regional LightGBM - LOADED")
                except Exception as e:
                    print(f"❌ MODEL 2: {region.title()} Regional - Failed")
        
        if self.regional_models:
            self.models_loaded['model2'] = True
        
        # Load Model 3: Seasonal models
        self.seasonal_models = {}
        seasons = ['rabi', 'kharif', 'zaid']
        for season in seasons:
            try:
                model_data = joblib.load(f'ML models/{season}_model.pkl')
                self.seasonal_models[season] = model_data
                print(f"✅ MODEL 3: {season.title()} Seasonal LightGBM - LOADED")
            except Exception as e:
                print(f"❌ MODEL 3: {season.title()} Seasonal - Not trained yet")
        
        if self.seasonal_models:
            self.models_loaded['model3'] = True
    
    def predict_model1(self, features):
        """Use general high-accuracy model"""
        if not self.models_loaded['model1']:
            return "Model 1 not available"
        
        try:
            feature_array = [[
                features['N'], features['P'], features['K'],
                features['Temperature'], features['Humidity'], features['pH'],
                features['Rainfall'], features.get('Moisture', 20), 0
            ]]
            
            features_scaled = self.scaler1.transform(feature_array)
            prediction = self.model1.predict(features_scaled)[0]
            crop_name = self.encoder1.inverse_transform([prediction])[0]
            
            return {
                'crop': crop_name,
                'confidence': 'High (LightGBM trained)',
                'model_type': 'High-Accuracy General LightGBM Model'
            }
        except Exception as e:
            return f"Model 1 error: {e}"
    
    def predict_model2(self, features):
        """Use regional specialized models"""
        if not self.models_loaded['model2']:
            return "Model 2 not available"
        
        region = features.get('region', 'punjab').lower()
        if region not in self.regional_models:
            return f"No regional model for {region}"
        
        try:
            model_data = self.regional_models[region]
            model = model_data['model']
            
            feature_array = [[
                features['N'], features['P'], features['K'], features['pH'],
                features['Temperature'], features['Humidity'], features['Rainfall'], 
                features['Moisture']
            ]]
            
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0]
            
            if 'encoder' in model_data:
                crop_name = model_data['encoder'].inverse_transform([prediction])[0]
            else:
                crop_name = f"Crop_{prediction}"
            
            confidence = np.max(probability)
            
            return {
                'crop': crop_name,
                'confidence': f"{confidence:.1%}",
                'model_type': f'{region.title()} Regional LightGBM Model'
            }
        except Exception as e:
            return f"Model 2 error: {e}"
    
    def predict_model3(self, features):
        """Use seasonal specialized models"""
        if not self.models_loaded['model3']:
            return "Model 3 not available"
        
        season = features.get('season', 'rabi').lower()
        if season not in self.seasonal_models:
            return f"No seasonal model for {season}"
        
        try:
            model_data = self.seasonal_models[season]
            model = model_data['model']
            encoder = model_data['encoder']
            
            feature_array = [[
                features['N'], features['P'], features['K'], features['pH'],
                features['Temperature'], features['Humidity'], features['Rainfall'], 
                features['Moisture']
            ]]
            
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0]
            crop_name = encoder.inverse_transform([prediction])[0]
            confidence = np.max(probability)
            
            return {
                'crop': crop_name,
                'confidence': f"{confidence:.1%}",
                'model_type': f'{season.title()} Seasonal LightGBM Model'
            }
        except Exception as e:
            return f"Model 3 error: {e}"
    
    def predict_all_models(self, features):
        """Get predictions from all 3 models"""
        print(f"\n🎯 GETTING PREDICTIONS FROM 3 LightGBM MODELS")
        print("=" * 40)
        
        results = {
            'model1_general': self.predict_model1(features),
            'model2_regional': self.predict_model2(features),
            'model3_seasonal': self.predict_model3(features)
        }
        
        return results

def main():
    system = FinalThreeModelSystem()
    
    print(f"\n📊 SYSTEM SUMMARY:")
    print(f"   Model 1 (General): {'✅ LOADED' if system.models_loaded['model1'] else '❌ MISSING'}")
    print(f"   Model 2 (Regional): {'✅ LOADED' if system.models_loaded['model2'] else '❌ MISSING'}")
    print(f"   Model 3 (Seasonal): {'✅ LOADED' if system.models_loaded['model3'] else '❌ MISSING'}")
    
    test_features = {
        'N': 100, 'P': 50, 'K': 40, 'pH': 7.0,
        'Temperature': 25, 'Humidity': 60, 'Rainfall': 100, 'Moisture': 20,
        'region': 'Punjab', 'season': 'Rabi', 'SoilType': 'Clay', 'Irrigation': 'Irrigated'
    }
    
    results = system.predict_all_models(test_features)
    
    print(f"\n🧪 PREDICTION RESULTS:")
    for model_name, result in results.items():
        print(f"\n   {model_name.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"      {key}: {value}")
        else:
            print(f"      {result}")

if __name__ == "__main__":
    main()

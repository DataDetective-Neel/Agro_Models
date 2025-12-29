# ML Model Original - Archive

This folder contains all the original ML models (classification and regression), datasets, and training scripts for the crop prediction system.

## 📁 Contents

### Training Scripts (.py files)
- **train_improved_models.py** - Enhanced regional crop prediction models with feature engineering
- **train_regional_models.py** - Region-specific models (Punjab, Haryana, Rajasthan)
- **train_seasonal_specialized.py** - Season-based models (Rabi, Kharif, Zaid)
- **final_three_model_system.py** - Combined 3-model prediction system
- **data_analysis.py** - Data analysis utilities

### Legacy Models (legacy/ folder)
- **train_models.py** - Original training script for classification and regression models
- **generate_dataset.py** - Synthetic dataset generation script
- **app.py** - Legacy Flask application

### Datasets (data/ folder)
#### Processed Data:
- `general_data.csv` - Combined processed dataset
- `haryana_data.csv` - Haryana-specific crop data
- `punjab_data.csv` - Punjab-specific crop data
- `rajasthan_data.csv` - Rajasthan-specific crop data

#### Raw Data:
- `combined_dataset.csv` - Original combined dataset

### Trained Models (ML models/ folder)
#### Classification Models:
- `crop_classifier_model.pkl` - Main crop classification model
- `haryana_model.pkl` - Haryana regional classifier
- `punjab_model.pkl` - Punjab regional classifier
- `rajasthan_model.pkl` - Rajasthan regional classifier
- `haryana_improved_model.pkl` - Enhanced Haryana model
- `punjab_improved_model.pkl` - Enhanced Punjab model
- `rajasthan_improved_model.pkl` - Enhanced Rajasthan model

#### Seasonal Models:
- `kharif_model.pkl` - Kharif season model
- `rabi_model.pkl` - Rabi season model
- `zaid_model.pkl` - Zaid season model

#### Regression Model:
- `soil_moisture_model.pkl` - Soil moisture prediction model

#### Supporting Files:
- `label_encoder.pkl` - Label encoding for crops
- `scaler_clf.pkl` - Feature scaler for classification
- `scaler_reg.pkl` - Feature scaler for regression
- `crop_features.pkl` - Crop feature definitions
- `moisture_features.pkl` - Moisture feature definitions
- `model_metadata.pkl` - Model metadata and information

## 📊 Model Types

### Classification Models
These models predict the best crop to grow based on environmental and soil conditions:
- Regional models (Punjab, Haryana, Rajasthan specific)
- Seasonal models (Rabi, Kharif, Zaid specific)
- Improved models with enhanced feature engineering

### Regression Model
- Soil moisture prediction based on environmental factors

## 🗓️ Moved On
**Date:** December 27, 2025
**From:** `a:\website\crop_system\`
**To:** `a:\website\ML_Model_Original\`

## 📝 Notes
All ML models, training scripts, and datasets have been consolidated into this folder for better organization and archival purposes.

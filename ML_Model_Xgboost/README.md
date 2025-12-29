# ML Model XGBoost - Enhanced Gradient Boosting Implementation

This folder contains XGBoost-based implementations of all crop prediction and soil moisture models, converted from the original Random Forest models.

## 📁 Contents

### Training Scripts (.py files)
- **train_improved_models.py** - Enhanced regional crop prediction models with XGBoost and feature engineering
- **train_regional_models.py** - Region-specific XGBoost models (Punjab, Haryana, Rajasthan)
- **train_seasonal_specialized.py** - Season-based XGBoost models (Rabi, Kharif, Zaid)
- **final_three_model_system.py** - Combined 3-model XGBoost prediction system
- **train_models.py** - Legacy XGBoost training script for classification and regression models

### Datasets (data/ folder)
#### Processed Data:
- `general_data.csv` - Combined processed dataset
- `haryana_data.csv` - Haryana-specific crop data
- `punjab_data.csv` - Punjab-specific crop data
- `rajasthan_data.csv` - Rajasthan-specific crop data

#### Raw Data:
- `combined_dataset.csv` - Original combined dataset

### Trained Models (ML models/ folder)
Models will be generated after running training scripts:

#### Classification Models:
- `crop_classifier_model.pkl` - Main crop classification XGBoost model
- `haryana_model.pkl` - Haryana regional XGBoost classifier
- `punjab_model.pkl` - Punjab regional XGBoost classifier
- `rajasthan_model.pkl` - Rajasthan regional XGBoost classifier
- `haryana_improved_model.pkl` - Enhanced Haryana XGBoost model
- `punjab_improved_model.pkl` - Enhanced Punjab XGBoost model
- `rajasthan_improved_model.pkl` - Enhanced Rajasthan XGBoost model

#### Seasonal Models:
- `kharif_model.pkl` - Kharif season XGBoost model
- `rabi_model.pkl` - Rabi season XGBoost model
- `zaid_model.pkl` - Zaid season XGBoost model

#### Regression Model:
- `soil_moisture_model.pkl` - Soil moisture prediction XGBoost model

#### Supporting Files:
- `label_encoder.pkl` - Label encoding for crops
- `scaler_clf.pkl` - Feature scaler for classification
- `scaler_reg.pkl` - Feature scaler for regression
- `crop_features.pkl` - Crop feature definitions
- `moisture_features.pkl` - Moisture feature definitions
- `model_metadata.pkl` - Model metadata and information

## 🚀 XGBoost Implementation Details

### Why XGBoost?

**XGBoost (eXtreme Gradient Boosting)** offers several advantages:
- **Higher Accuracy**: Sequential boosting often outperforms bagging approaches
- **Better Regularization**: Built-in L1/L2 regularization prevents overfitting
- **Feature Importance**: More reliable feature importance metrics
- **Efficient Training**: Optimized for speed with parallel processing
- **Handle Missing Data**: Native support for missing values

### Key Parameters Used

#### For Classification Models:
```python
XGBClassifier(
    n_estimators=200-300,    # Number of boosting rounds
    max_depth=6-8,           # Tree depth (shallower than RF)
    learning_rate=0.05-0.1,  # Shrinkage for regularization
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    gamma=0.1,               # Min split loss
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    eval_metric='mlogloss'
)
```

#### For Regression Models:
```python
XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

## 📊 Model Types

### Classification Models
These XGBoost models predict the best crop to grow based on environmental and soil conditions:
- **Regional models**: Punjab, Haryana, Rajasthan specific
- **Seasonal models**: Rabi, Kharif, Zaid specific
- **Improved models**: Enhanced with feature engineering

### Regression Model
- **Soil moisture prediction** based on environmental factors using XGBoost Regressor

## 🔄 Conversion from Random Forest

All models have been systematically converted from Random Forest to XGBoost:
- Maintained the same feature engineering pipeline
- Adapted hyperparameters for optimal XGBoost performance
- Preserved data preprocessing and scaling
- Same prediction interface for easy integration

## 📝 Training the Models

To train all models, run the training scripts in order:

```bash
# Train legacy models (classification + regression)
python train_models.py

# Train regional models
python train_regional_models.py

# Train improved regional models
python train_improved_models.py

# Train seasonal models
python train_seasonal_specialized.py

# Test the complete system
python final_three_model_system.py
```

## 🗓️ Created On
**Date:** December 27, 2025  
**Source:** Converted from `ML_Model_Original`  
**Method:** Random Forest → XGBoost migration

## 📈 Performance Expectations

XGBoost models typically achieve:
- **Improved accuracy** compared to Random Forest (1-3% gain)
- **Better generalization** through regularization
- **Faster inference** for production deployment
- **More interpretable** feature importance scores

## 🔗 Related Documentation

See [COMPARISON.md](COMPARISON.md) for detailed comparison between Random Forest and XGBoost implementations.

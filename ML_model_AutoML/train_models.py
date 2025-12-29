# train_models.py - AutoML Version (AutoGluon)
import pandas as pd
from autogluon.tabular import TabularPredictor
import os

print("STARTING AutoML MODEL TRAINING (AutoGluon)")
print("=" * 60)

# Create models directory  
os.makedirs('ML models', exist_ok=True)

# Load dataset
print("\nLOADING DATASET...")
df = pd.read_csv('data/raw/combined_dataset.csv')
print(f"Dataset loaded: {df.shape}")
print(f"Crops: {df['Crop'].unique()}")

# AutoGluon requires minimal preprocessing!
# Split data
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Crop'])

print(f"\nTrain set: {train_df.shape}")
print(f"Test set: {test_df.shape}")

# Train AutoML Crop Classifier
print("\nTRAINING AutoGluon CROP MODEL...")
print("AutoGluon will automatically:")
print("  - Test multiple models (LightGBM, CatBoost, XGBoost, Neural Nets, etc.)")
print("  - Perform hyperparameter tuning")
print("  - Create ensemble models")
print("  - Handle feature engineering")

crop_predictor = TabularPredictor(
    label='Crop',
    path='ML models/autogluon_crop_model',
    eval_metric='accuracy',
    problem_type='multiclass'
).fit(
    train_df,
    time_limit=300,  # 5 minutes - adjust as needed
    presets='best_quality',  # Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    verbosity=2
)

# Evaluate
print("\nEVALUATING AutoML MODEL...")
predictions = crop_predictor.predict(test_df)
accuracy = (predictions == test_df['Crop']).mean()

print(f"\n🎯 CROP MODEL PERFORMANCE:")
print(f"Overall Accuracy: {accuracy:.2%}")

# Show leaderboard
print("\n📊 MODEL LEADERBOARD:")
leaderboard = crop_predictor.leaderboard(test_df)
print(leaderboard)

# Feature importance
print("\n📈 FEATURE IMPORTANCE:")
importance = crop_predictor.feature_importance(test_df)
print(importance)

print("\n✅ AutoML MODEL TRAINING COMPLETED!")
print("Models saved in 'ML models/autogluon_crop_model/' directory")
print("\nBest model will be automatically selected for predictions!")
print("AutoML tested multiple algorithms and created the optimal ensemble!")

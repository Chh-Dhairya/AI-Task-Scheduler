from data_loader import load_dataset
from preprocess import DataPreprocessor
from habit_model import HabitModel

DATA_PATH = "C:\\Users\\user\\OneDrive\\Desktop\\task_scheduler\\activities_of_daily_living_fixed.csv"

df = load_dataset(DATA_PATH)
preprocessor = DataPreprocessor()
df = preprocessor.fit_transform(df)

model = HabitModel()
model.train(df)
model.save("habit_model.pkl")

# Accuracy check
correct, total = 0, 0
for _, row in df.iterrows():
    pred = model.predict_bucket(row['Person_Encoded'], row['Activity_Encoded'])
    if pred == row['TimeBucket']:
        correct += 1
    total += 1

print(f"\nTraining accuracy: {round(correct / total * 100, 1)}%")
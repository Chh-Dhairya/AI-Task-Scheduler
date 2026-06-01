from data_loader import load_dataset
from preprocess import DataPreprocessor
from duration_model import DurationModel

df = load_dataset("C:\\Users\\user\\OneDrive\\Desktop\\activities_of_daily_living_fixed.csv")

df["DayOfWeek"] = df["Start"].dt.dayofweek

preprocessor = DataPreprocessor()
df = preprocessor.fit_transform(df)

duration_model = DurationModel()
duration_model.train(df)

duration_model.save("duration_model.pkl")

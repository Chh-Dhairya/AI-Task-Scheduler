import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_time_bucket(hour):
    if 5 <= hour < 9:    return 0   # early morning
    elif 9 <= hour < 13: return 1   # morning
    elif 13 <= hour < 17: return 2  # afternoon
    elif 17 <= hour < 21: return 3  # evening
    elif 21 <= hour <= 23: return 4 # night
    else: return 4                  # 0-4 AM → also night (merge with bucket 4)


class DataPreprocessor:
    def __init__(self):
        self.person_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()

    def fit_transform(self, df):
        df = df.copy()
        df["Person_Encoded"] = self.person_encoder.fit_transform(df["Person ID"])
        df["Activity_Encoded"] = self.activity_encoder.fit_transform(df["Activity Type"])
        df["TimeBucket"] = df["Hour"].apply(get_time_bucket)
        df["DayOfWeek"] = df["Start"].dt.dayofweek
        return df

    def get_metadata(self):
        return (
            len(self.person_encoder.classes_),
            len(self.activity_encoder.classes_)
        )

    def transform_activity(self, activity):
        return self.activity_encoder.transform([activity])[0]


def build_sequences(df, sequence_length=5):
    """
    FIXED: target label is buckets[i + sequence_length], not i + sequence_length - 1.
    Also includes day-of-week of the TARGET activity as a feature.
    """
    X_person, X_activity_seq, X_dow, y_bucket = [], [], [], []

    for person_id in df["Person_Encoded"].unique():
        person_data = df[df["Person_Encoded"] == person_id].reset_index(drop=True)
        activities = person_data["Activity_Encoded"].values
        buckets    = person_data["TimeBucket"].values
        dow        = person_data["DayOfWeek"].values

        for i in range(len(activities) - sequence_length):
            X_person.append(person_id)
            X_activity_seq.append(activities[i : i + sequence_length])
            X_dow.append(dow[i + sequence_length])          # day of TARGET activity
            y_bucket.append(buckets[i + sequence_length])   # label = NEXT activity's bucket

    return (
        np.array(X_person),
        np.array(X_activity_seq),
        np.array(X_dow),
        np.array(y_bucket)
    )
import pandas as pd
from sklearn.model_selection import train_test_split

# Import and clean data
channels = pd.read_csv("data/channels.csv").rename(columns={"followers": "subscribers"})
channels.country = channels.country.fillna("Unknown")

# Select features and target
features = ["category_name", "country", "join_date", "videos"]
X = channels[features]
y = channels.subscribers

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
print(X_train, X_test, y_train, y_test)

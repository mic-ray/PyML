import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score


# Import and clean data
channels = pd.read_csv("data/channels.csv").rename(columns={"followers": "subscribers"})
# Filter out rows, where a join date is not provided
channels = channels.loc[channels.join_date.notnull()]

# Converts a date string to a timestamp
def convert_date_to_timestamp(date):
    return datetime.fromisoformat(str(date)).timestamp()


# Convert date strings to a float value
channels.join_date = channels.join_date.apply(convert_date_to_timestamp)

# Select (only numeric) features and target
features = ["join_date", "videos"]
X = channels[features]
y = channels.subscribers

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Create a model and fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicate values and score model
y_pred = model.predict(X_test)
print(explained_variance_score(y_test, y_pred))

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('dataset1.csv')
df = df.astype('float64')

X = df.drop(columns=['ritmi'])
y = df['ritmi']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train RandomForestClassifier on raw data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the trained model
pickle.dump(rf, open('rf_model.pkl', 'wb'))

print("RandomForest model trained and saved successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("mobile_phones.csv")   


# Preprocess the data
# Handle missing values (example)
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in ['battery', 'camera', 'display', 'processor', 'Price Class']:
    data[col] = le.fit_transform(data[col])

# Scale numerical features
scaler = StandardScaler()
data['price'] = scaler.fit_transform(data['price'].values.reshape(-1, 1))

# Split into features and target
X = data.drop('rating', axis=1)
y = data['rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model (example: Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions   

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)   

rmse = mse**0.5
print("RMSE:", rmse)
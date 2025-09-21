# Import Libraries
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import joblib # Import joblib
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv("Data\Telco_Customer_Churn.csv")

# Create a backup of the Dataframe
data_original = data.copy()

# Converting Total Charges
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = "coerce")

data.dropna(inplace=True)
data["Churn"].replace(to_replace="Yes", value =1, inplace=True)
data["Churn"].replace(to_replace="No", value =0, inplace=True)

df2 = data.iloc[:,1:]
df_dummies = pd.get_dummies(df2)

def scale_dataset(df, oversample = False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x,y = ros.fit_resample(x,y)

    df1 = np.hstack((x,np.reshape(y, (-1,1))))

    return df1, x,y

# Training and Testing Dataset
train, valid, test = np.split(df_dummies.sample(frac=1),[int(0.6*len(df_dummies)), int(0.8*len(df_dummies))])

train_data, x_train, y_train = scale_dataset(train, oversample = True)
valid_data, x_valid, y_valid = scale_dataset(valid, oversample = False)
test_data, x_test, y_test = scale_dataset(test, oversample = False)

# Initial Model Training

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# Remove invalid arguments for GradientBoostingClassifier fit method
model.fit(x_train, y_train)

# Save the model using joblib

joblib.dump(model, "tele_churn.h5")

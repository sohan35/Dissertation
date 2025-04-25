from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import os
import random
import folium

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv("sy.csv", encoding='latin1')
df['Geo_Location'] = df['Geo_Location'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.strip()
df[['Latitude', 'Longitude']] = df['Geo_Location'].str.split(',', expand=True)
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df = df.dropna(subset=["Latitude", "Longitude"])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month_name()
df['Crime_Location_Type'] = df['Crime_Location_Type'].astype(str).str.strip()
df['Crime_Location_Encoded'] = LabelEncoder().fit_transform(df['Crime_Location_Type'])
df['Head'] = df['Head'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
df['Head_clean'] = df['Head'].str.lower().str.strip()

# One-hot encode 'Head_clean' and combine with lat/lon
crime_type_dummies = pd.get_dummies(df['Head_clean'], prefix='crime')
features = pd.concat([df[['Latitude', 'Longitude']], crime_type_dummies], axis=1)
labels = df['Crime_Location_Encoded']

# Data Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Hyperparameter Tuning (using full dataset with cross-validation)
best_k = 3
best_score = 0
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, features_scaled, labels, cv=5)
    avg_score = scores.mean()
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

# Final Model Training on full dataset
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(features_scaled, labels)
model_accuracy = best_score  # since we didn't split, use CV score as accuracy

# Decoder
label_decoder = dict(zip(df['Crime_Location_Encoded'], df['Crime_Location_Type']))

# Station colors for map
station_colors = {
    "Mapusa_Ps": "blue",
    "Anjuna_Ps": "red",
    "Colvale_PS": "purple"
}
default_colors = ["red", "orange", "cadetblue", "darkred", "lightgreen", "darkpurple"]

def generate_map(filtered_df):
    if filtered_df.empty:
        return None
    crime_map = folium.Map(location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()], zoom_start=12)
    for _, row in filtered_df.iterrows():
        station = row["Police Station"]
        color = station_colors.get(station, random.choice(default_colors))
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{station}: {row['Crime_Location_Type']}"
        ).add_to(crime_map)
    map_path = "crime_map.html"
    map_file = os.path.join("templates", map_path)
    crime_map.save(map_file)
    return f"/map/{map_path}"

@app.route("/map/<path:filename>")
def get_map(filename):
    return send_from_directory("templates", filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    available_crimes = sorted(df['Head'].dropna().unique())

    if request.method == 'POST':
        user_input = request.form['crime_type'].lower().strip()
        matched_crime = None
        for crime in df['Head_clean'].unique():
            if user_input in crime:
                matched_crime = crime
                break

        if matched_crime:
            filtered_crime_df = df[df['Head_clean'] == matched_crime]
            matched_display = filtered_crime_df['Head'].iloc[0]

            special_inhouse_crimes = ["theft (hbt day)", "theft (hbt night)", "theft (house theft)"]
            if matched_display.lower() in special_inhouse_crimes:
                filtered_crime_df['Crime_Location_Type'] = "InHouse"

            top_station = filtered_crime_df['Police Station'].value_counts().idxmax()
            top_station_cases = filtered_crime_df['Police Station'].value_counts().max()
            top_location = filtered_crime_df['Crime_Location_Type'].value_counts().idxmax()
            top_location_cases = filtered_crime_df['Crime_Location_Type'].value_counts().max()

            allocation_info = []

            for station in df['Police Station'].unique():
                station_df = filtered_crime_df[filtered_crime_df['Police Station'] == station]
                station_result = {
                    'station': station,
                    'allocation': [],
                    'peak_months': []
                }

                if station_df.empty:
                    station_result['allocation'].append("No crimes of this type reported. Allocate minimum patrol if needed.")
                else:
                    location_counts = station_df['Crime_Location_Type'].value_counts()
                    total_crimes = location_counts.sum()
                    top_locations = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:4])

                    for location_type, count in top_locations.items():
                        location_type_clean = location_type.strip().lower()
                        if location_type_clean == 'inhouse' and matched_display.lower() not in special_inhouse_crimes:
                            station_result['allocation'].append("InHouse: Investigate on a case-by-case basis. No direct deployment.")
                        else:
                            if count <= 10:
                                station_result['allocation'].append(f"{location_type}: Low crime ({count} cases) – Deploy 2 police officers.")
                            elif 11 <= count <= 30:
                                station_result['allocation'].append(f"{location_type}: Medium crime ({count} cases) – Deploy 4 police officers.")
                            else:
                                station_result['allocation'].append(f"{location_type}: High crime ({count} cases) – Deploy 6 police officers.")

                    top_months = station_df['Month'].value_counts().head(3).index.tolist()
                    station_result['peak_months'] = top_months

                allocation_info.append(station_result)

            # Predict with KNN
            mean_coords = filtered_crime_df[['Latitude', 'Longitude']].mean()
            crime_dummy = pd.get_dummies([matched_crime], prefix='crime')
            crime_features = pd.DataFrame(columns=crime_type_dummies.columns)
            crime_features.loc[0] = 0
            crime_features.update(crime_dummy)

            input_features = pd.DataFrame([[mean_coords['Latitude'], mean_coords['Longitude']]], columns=['Latitude', 'Longitude'])
            input_full = pd.concat([input_features, crime_features], axis=1).fillna(0)
            input_scaled = scaler.transform(input_full)
            predicted_class = knn.predict(input_scaled)[0]
            predicted_location = label_decoder.get(predicted_class, "Unknown")

            map_path = generate_map(filtered_crime_df)

            results = {
                'matched_display': matched_display,
                'top_station': top_station,
                'top_station_cases': top_station_cases,
                'top_location': top_location,
                'top_location_cases': top_location_cases,
                'allocation_info': allocation_info,
                'map_path': map_path,
                'predicted_location': predicted_location,
                'model_accuracy': f"{model_accuracy * 100:.2f}%",
                'best_k': best_k
            }
        else:
            results = {'error': "No matching crime type found."}

    return render_template('index.html', crimes=available_crimes, results=results)

if __name__ == '__main__':
    app.run(debug=True)
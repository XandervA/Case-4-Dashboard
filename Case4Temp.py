#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st
st.set_page_config(page_title="Case 4 - Stand van het klimaat")

# Importing data
uitstoot_1990_2022 = pd.read_csv("table_tabel-812eb65c-5599-431c-82ef-1e3d36541c49.csv", index_col=0, decimal=',')
uitstoot_1990_2022.drop(uitstoot_1990_2022.tail(1).index, inplace=True)
gem_temp = pd.read_excel("Gemiddelde temperatuur (per decennium) - Amersfoort.xlsx")
zonuren = pd.read_excel("Zonuren per dec.xlsx")
df12 = pd.read_csv("table_tabel-35e3e1a4-9240-4e4c-97ad-a983c0e2978d.csv")
df12.rename(columns={"Personenauto's - benzine (megaton CO2-equivalent)": 'Personenauto(benzine)',
                     "Personenauto's - diesel (megaton CO2-equivalent)": 'Personenauto(diesel)',
                     'Vrachtvoertuigen (megaton CO2-equivalent)': 'Vrachtvoertuigen',
                     'Mobiele werktuigen (megaton CO2-equivalent)': 'Mobiele werktuigen',
                     'Overig verkeer en vervoer (megaton CO2-equivalent)': 'Overig'}, inplace=True)
df12[['Personenauto(benzine)', 'Personenauto(diesel)', 'Vrachtvoertuigen', 'Mobiele werktuigen', 'Overig']] = df12[
    ['Personenauto(benzine)', 'Personenauto(diesel)', 'Vrachtvoertuigen', 'Mobiele werktuigen', 'Overig']].apply(lambda x: x.str.replace(',', '.').astype(float))
df15 = pd.read_excel("CO2_wereld.xlsx")
df16 = pd.read_csv("annual-co2-emissions-per-country.csv")
df16 = df16[(df16['Year'] < 1750) | (df16['Year'] > 2000)]
brics = ['Brazil', 'India', 'China','United States', 'Netherlands', 'United Kingdom']
df16 = df16[df16['Entity'].isin(brics)]


# Chart 1
fig_uitstoot = px.line(uitstoot_1990_2022, title='Uitstoot per Categorie over de Jaren',
                       color_discrete_sequence=["yellow", "blue", "pink", "skyblue","red","orange"])
fig_uitstoot.update_layout(legend_title_text='Sector', xaxis_title_text='Jaar', yaxis_title_text='Uitstoot (Megaton CO2-equivalent)')

fig_uitstoot.data[0].name = 'Industrie'
fig_uitstoot.data[1].name = 'Elektriciteit'
fig_uitstoot.data[2].name = 'Mobiliteit'
fig_uitstoot.data[3].name = 'Gebouwde omgeving'
fig_uitstoot.data[4].name = 'Landbouw'
fig_uitstoot.data[5].name = 'Landgebruik'

# Chart 2 & 3 (OLS)
def predict_co2_levels(X, y, future_years, saturation_point=None):
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict CO2 levels for the years 2023-2050
    future_predictions = model.predict(future_years)
    
    # Introduce a saturation point to limit decrease
    if saturation_point is not None:
        saturation_point = np.repeat(saturation_point, len(future_predictions) // len(saturation_point) + 1)[:len(future_predictions)]
        saturation_mask = future_predictions < saturation_point
        future_predictions[saturation_mask] = saturation_point[saturation_mask]
    
    return future_predictions

# Convert the year values to integers in the index
uitstoot_1990_2022.index = uitstoot_1990_2022.index.astype(int)

# Define the recent years for training in both versions
recent_years_all = uitstoot_1990_2022.index >= 1990
recent_years_recent = uitstoot_1990_2022.index >= 2010

# Calculate the saturation point as a 95% reduction from the original value in 1990
original_values_1990 = uitstoot_1990_2022.iloc[0].values
saturation_point = 0.05 * original_values_1990

# Version 1: Training on all available years
X_all = np.array(uitstoot_1990_2022.index[recent_years_all]).reshape(-1, 1)
predictions_all = {industry: predict_co2_levels(X_all, uitstoot_1990_2022[industry][recent_years_all], np.arange(2023, 2051).reshape(-1, 1), saturation_point=saturation_point) for industry in uitstoot_1990_2022.columns}

# Version 2: Training on recent years only
X_recent = np.array(uitstoot_1990_2022.index[recent_years_recent]).reshape(-1, 1)
predictions_recent = {industry: predict_co2_levels(X_recent, uitstoot_1990_2022[industry][recent_years_recent], np.arange(2023, 2051).reshape(-1, 1), saturation_point=saturation_point) for industry in uitstoot_1990_2022.columns}

# Create DataFrames for predictions
df_predictions_all = pd.DataFrame(predictions_all, index=pd.date_range('2023-01-01', periods=28, freq='Y'))
df_predictions_recent = pd.DataFrame(predictions_recent, index=pd.date_range('2023-01-01', periods=28, freq='Y'))

# Plotting using Plotly Express
fig_pred_full = px.line(df_predictions_all, x=df_predictions_all.index, y=df_predictions_all.columns,
                  labels={'index': 'Jaar', 'value': 'Uitstoot (Megaton CO2-equivalent)'},
                  title='Voorspelling van CO2-niveaus tot 2050 per industrie (met gebruik van gegevens van 1990-2022)',
                  color_discrete_sequence=["yellow", "blue", "pink", "skyblue","red","orange"])
fig_pred_full.update_layout(legend_title_text='Sector')
fig_pred_full.data[0].name = 'Industrie'
fig_pred_full.data[1].name = 'Elektriciteit'
fig_pred_full.data[2].name = 'Mobiliteit'
fig_pred_full.data[3].name = 'Gebouwde omgeving'
fig_pred_full.data[4].name = 'Landbouw'
fig_pred_full.data[5].name = 'Landgebruik'

fig_pred_recent = px.line(df_predictions_recent, x=df_predictions_recent.index, y=df_predictions_recent.columns,
                     labels={'index': 'Jaar', 'value': 'Uitstoot (Megaton CO2-equivalent)'},
                     title='Voorspelling van CO2-niveaus tot 2050 per industrie (met gebruik van gegevens van 2010-2022)',
                     color_discrete_sequence=["yellow", "blue", "pink", "skyblue","red","orange"])
fig_pred_recent.update_layout(legend_title_text='Sector')
fig_pred_recent.data[0].name = 'Industrie'
fig_pred_recent.data[1].name = 'Elektriciteit'
fig_pred_recent.data[2].name = 'Mobiliteit'
fig_pred_recent.data[3].name = 'Gebouwde omgeving'
fig_pred_recent.data[4].name = 'Landbouw'
fig_pred_recent.data[5].name = 'Landgebruik'

# Chart 3 & 4 (Zonuren & Temp)
fig_temp = px.line(gem_temp, x='periode', y='waarde', title='Gemiddelde temperatuur (per decennium)', color_discrete_sequence=["crimson"])
y_axis_range = [8.8, 10.8]  # Adjust these values as needed

fig_temp.update_layout(
    xaxis_title='Jaar',
    yaxis=dict(
        range=y_axis_range,
        title='Graden Celcius',
        dtick=0.4
    )
)

fig_temp.update_traces(mode='lines+markers')

fig_zon = px.line(zonuren, x='Jaar', y='Totale zonuren per jaar per decennium', title='Totale zonuren per jaar per decennium', color_discrete_sequence=["orange"])
fig_zon.update_layout(yaxis_title='Zonuren', yaxis=dict(range=[1450, 1850], dtick=50))
fig_zon.update_traces(mode='lines+markers')

# Plot 5 (mobiliteit uitstoot)
fig_auto = px.bar(df12,x="\xa0", y=["Personenauto(benzine)","Personenauto(diesel)","Vrachtvoertuigen","Mobiele werktuigen","Overig"],
                                      title="Uitstoot door sector Mobiliteit", color_discrete_sequence=["blue", "crimson", "green", "purple","orange"])
fig_auto.update_layout(
    yaxis_title='Uitstoot (Megaton CO2-equivalent)',
    xaxis_title='Jaar',
    legend_title_text='Bronnen')

# Plot 6 (Wereld data)
fig_wereld = px.line(df15, x="Jaar", y='CO2', 
             title="CO2 uitstoot wereldwijd", markers=True)
fig_wereld.update_layout(yaxis_title='Uitstoot CO2 in Miljarden M^3')


# Plot 7 (Geselecteerde landen)
fig_select = px.line(df16, x="Year", y="Annual CO₂ emissions", color = "Entity", title = 'CO2 uitstoot geselecteerde landen',
              log_y=True, color_discrete_sequence=["yellow", "blue", "pink", "skyblue","red","orange"])
fig_select.update_layout(
    yaxis_title='Uitstoot CO2 in M^3',
    xaxis_title='Jaar',
    legend_title_text='Landen')


# Streamlit section
st.title("Case 4 - Stand van het Klimaat")
st.caption("By Xander van Altena and Salah Bentaher")

st.subheader("Huidige situatie")
st.plotly_chart(fig_uitstoot, use_container_width=True)
st.divider()

st.subheader("Regressieanalyse")
st.plotly_chart(fig_pred_full, use_container_width=True)
st.plotly_chart(fig_pred_recent, use_container_width=True)
st.divider()

st.subheader("In welke regio's zijn verbeteringen cruciaal")
st.markdown("CO2-uitstoot Industrie, Energie, Afval en Water (2021)")
st.image('Uitstoot_gebouwde_omgeving.png', use_column_width=True)
st.markdown("CO2-uitstoot Landbouw (2021)")
st.image('Uitstoot landbouw.png', use_column_width=True)
st.caption("Bron: Centraal Bureau voor de Statistiek (CBS), RIVM/Emissieregistratie")
st.divider()

st.subheader("Groei auto's hindert CO2-reductie ondanks nieuwe regulaties en milieuvriendelijkere voertuigen")
st.markdown("De hoeveelheid auto's is gegroeid met 2,6 miljoen sinds het jaar 2000.")
st.plotly_chart(fig_auto, use_container_width=True)

st.subheader("Gevolgen voor het climaat")
st.plotly_chart(fig_temp, use_container_width=True)
st.plotly_chart(fig_zon, use_container_width=True)
st.caption("Bron: KNMI")
st.divider()

st.subheader("Wereldbeeld")
st.plotly_chart(fig_wereld, use_container_width=True)
st.plotly_chart(fig_select, use_container_width=True)
st.caption('Bron: Global Carbon Project')


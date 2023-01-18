from string import digits

import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import pandas as pd
import re

import seaborn as sns
from numpy import nan
from tqdm import tqdm
import numpy

dataframe = pd.read_csv("immobiliare.csv")
api = "AIzaSyBBh2tKOtB6mZ4BinYbVvWY0uBCsiuxGg8"


def get_coordinates(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json?address=" + address + "&key=" + api
    response = requests.get(url)
    json_data = json.loads(response.text)
    return json_data


def get_nearby_places(lat, lng, type):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + str(lat) + "," + str(
        lng) + "&radius=15000&type=" + type + "&key=" + api
    response = requests.get(url)
    json_data = json.loads(response.text)
    return json_data


def get_distance(lat1, lng1, lat2, lng2):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=" + str(lat1) + "," + str(
        lng1) + "&destinations=" + str(lat2) + "," + str(lng2) + "&key=" + api
    response = requests.get(url)
    json_data = json.loads(response.text)
    return json_data


def main():
    try:
        dataframe.drop(columns=["web-scraper-order", "web-scraper-start-url", "Proprietà", "Proprietà-href", "Nome"],
                   axis=1, inplace=True)
    except:
        pass
    print("Mettendo solo valori numerici")
    for index, row in tqdm(dataframe.iterrows()):
        dataframe.at[index, 'PostoAuto'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'PostoAuto']))
        dataframe.at[index, 'EfficenzaEnergetica'] = re.sub("[^0-9]", "",
                                                            str(dataframe.at[index, 'EfficenzaEnergetica']))
        dataframe.at[index, 'SpeseCondominiali'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'SpeseCondominiali']))
        dataframe.at[index, 'target'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'target']))
        dataframe.at[index, 'Superficie'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'Superficie']))
        dataframe.at[index, 'Bagni'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'Bagni']))

    dataframe.replace("", np.nan, inplace=True)
    index = -1
    dataframe["City"] = "City"
    addressos = dataframe["Indirizzo"].to_list()
    print("Aggiungendo distanze")
    for address in tqdm(addressos):
        index += 1
        city = str(address).split("\n")
        dataframe.at[index,"City"] = city[0]
        if float(dataframe.at[index, 'Superficie']) < 10 or (float(dataframe.at[index, 'Superficie']) > 4000):
            dataframe.drop(index, inplace=True)
            continue
        if str(dataframe.at[index, 'Stato']) == "nan":
            dataframe.drop(index, inplace=True)
            continue
        if str(dataframe.at[index, 'AltreCaratteristiche1']) == "nan":
            dataframe.drop(index, inplace=True)
            continue
        if str(dataframe.at[index, 'Piano']) == "nan":
            dataframe.drop(index, inplace=True)
            continue
        if str(dataframe.at[index, 'Indirizzo']) == "nan":
            dataframe.drop(index, inplace=True)
            continue
        if float(dataframe.at[index, 'EfficenzaEnergetica']) > 999:
            num = str(dataframe.at[index, 'EfficenzaEnergetica'])[:3]
            dataframe.at[index, 'EfficenzaEnergetica'] = float(num)
        if float(dataframe.at[index, 'Locali']) > 10:
            dataframe.drop(index, inplace=True)
            continue
        if "Appartamento, Intera proprieta" in str(dataframe.at[index, 'Tipologia']):
            dataframe.at[index, 'Tipologia'] = "Appartamento, Intera proprieta"
        if "Appartamento, Intera proprieta" in str(dataframe.at[index, 'Tipologia']):
            dataframe.at[index, 'Tipologia'] = "Appartamento, Intera proprieta"
        if "Appartamento, Intera proprieta" in str(dataframe.at[index, 'Tipologia']):
            dataframe.at[index, 'Tipologia'] = "Appartamento, Intera proprieta"
        if "Appartamento, Intera proprieta" in str(dataframe.at[index, 'Tipologia']):
            dataframe.at[index, 'Tipologia'] = "Appartamento, Intera proprieta"
        if str( dataframe.at[index, 'AnnoDiCostruzione']) != "nan":
            dataframe.at[index, 'AnnoDiCostruzione'] = float(dataframe.at[index, 'AnnoDiCostruzione']) * 1000
        if (float(dataframe.at[index, 'target']) < 5000) or (str(dataframe.at[index, 'target']) == "nan"):
            dataframe.drop(index, inplace=True)
            continue
        if float(dataframe.at[index, 'Bagni']) > 5:
            dataframe.at[index, 'Bagni'] = 5
        if 9 < float(dataframe.at[index, 'PostoAuto']) < 100:
            num = str(dataframe.at[index, 'PostoAuto'])[:1]
            dataframe.at[index, 'PostoAuto'] = float(num)
        if float(dataframe.at[index, 'PostoAuto']) > 100:
            num = str(dataframe.at[index, 'PostoAuto'])[:2]
            dataframe.at[index, 'PostoAuto'] = float(num)
    dataframe['SpeseCondominiali'].fillna(0, inplace=True)
    dataframe['PostoAuto'].fillna(0, inplace=True)
    dataframe['AnnoDiCostruzione'].fillna(dataframe["AnnoDiCostruzione"].mean(axis=0, skipna=True), inplace=True)
    dataframe['Bagni'].fillna(0, inplace=True)
    dataframe['Riscaldamento'].fillna("Autonomo, a radiatori, alimentato a metano", inplace=True)
    dataframe['Climatizzazione'].fillna("Non presente", inplace=True)
    dataframe['EfficenzaEnergetica'].fillna(175.0, inplace=True)
    try:
        dataframe.drop(["EfficenzaEnergetica","SpeseCondominiali","AltreCaratteristiche1"],
                   axis=1, inplace=True)
    except:
        pass
    datafraame = dataframe.astype({"Superficie": int, "Bagni": int,"PostoAuto": int,"target": int})
    datafraame.info()
    datafraame.to_csv("dataset.csv")

main()
import requests
import json
import pandas as pd
import re
from tqdm import tqdm
import numpy

dataframe = pd.read_excel("immobiliare.xlsx")
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
    dataframe.drop(columns=["web-scraper-order", "web-scraper-start-url", "Proprietà", "Proprietà-href", "Nome"],
                   axis=1, inplace=True)
    print("Mettendo solo valori numerici")
    for index, row in tqdm(dataframe.iterrows()):
        dataframe.at[index, 'PostoAuto'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'PostoAuto']))
        dataframe.at[index, 'EfficenzaEnergetica'] = re.sub("[^0-9]", "",
                                                            str(dataframe.at[index, 'EfficenzaEnergetica']))
        dataframe.at[index, 'SpeseCondominiali'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'SpeseCondominiali']))
        dataframe.at[index, 'Prezzo'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'Prezzo']))
        dataframe.at[index, 'Superficie'] = re.sub("[^0-9]", "", str(dataframe.at[index, 'Superficie']))

    index = -1
    addressos = dataframe["Indirizzo"].to_list()
    print("Aggiungendo distanze")
    for address in tqdm(addressos):
        index += 1
        city = str(address).split("\n")
        dataframe.at[index, "City"] = city[0]
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
        if float(dataframe.at[index, 'Locali']) > 5:
            dataframe.drop(index, inplace=True)
            continue
        if str(dataframe.at[index, 'AnnoDiCostruzione']) != "nan":
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

        coordinates = get_coordinates(address)
        lat = coordinates["results"][0]["geometry"]["location"]["lat"]
        lng = coordinates["results"][0]["geometry"]["location"]["lng"]
        places = get_nearby_places(lat, lng, "restaurant")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaRistorante" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "grocery_or_supermarket")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaSupermercato" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "movie_theater")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaCinema" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "university")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaUniversità" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "police")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaPolizia" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "bus_station")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaBus" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "pharmacy")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaFarmacia" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "primary_school")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaScuolaPrimaria" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "secondary_school")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaScuolaSecondaria" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "train_station")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaStazioneTreno" + str(c)] = distanza
        places = get_nearby_places(lat, lng, "dentist")
        c = 0
        for place in places["results"]:
            if "closed" in str(place["business_status"]).lower():
                continue
            elif c >= 2:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaDentista" + str(c)] = distanza
        print("Completato l'indirizzo:" + address)
        places = get_nearby_places(lat, lng, "highway")
        c = 0
        for place in places["results"]:
            if c >= 1:
                break
            else:
                c += 1
                distance = get_distance(lat, lng, place["geometry"]["location"]["lat"],
                                        place["geometry"]["location"]["lng"])
                dist1 = distance["rows"][0]["elements"][0]["distance"]["text"]
                if "km" in dist1.lower():
                    dist2 = dist1.replace(" km", "")
                    distanza = float(dist2) * 1000
                else:
                    dist2 = dist1.replace(" m", "")
                    distanza = float(dist2)
                dataframe["DistanzaAutostrada" + str(c)] = distanza
        print("Completato l'indirizzo:" + address)
    dataframe.to_excel("immobiliare.xlsx")
    dataframe.to_csv("immobiliare.csv")

    print("Completato!")


main()

import wget
import pandas as pd


if __name__ == '__main__':
    table = pd.read_csv("data/stations_spb.csv", index_col=1,
                           usecols=['STATION', 'STATION_ID'])
    stations = table['STATION_ID'].unique().tolist()
    print(len(stations))
    # stations = stations[35:68]

    stations_str = ""
    for i in stations:
        stations_str += str(i) + ','
    stations_str = stations_str[:-1]

    url = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-summary-of-the-day&dataTypes=LATITUDE,LONGITUDE,ELEVATION,NAME,TEMP,TEMP_ATTRIBUTES,DEWP,DEWP_ATTRIBUTES,SLP,SLP_ATTRIBUTES,STP,STP_ATTRIBUTES,VISIB,VISIB_ATTRIBUTES,WDSP,WDSP_ATTRIBUTES,MXSPD,GUST,MAX,MAX_ATTRIBUTES,MIN,MIN_ATTRIBUTES,PRCP,PRCP_ATTRIBUTES,SNDP,FRSHTT&stations=" + stations_str + "&startDate=2010-01-01&endDate=2020-01-02"
    wget.download(url, "data/stations_one_dim.csv")

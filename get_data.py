"""
The function downloads data from the www.ncei.noaa.gov website

Inputs
stations_file: The csv file from https://gis.ncdc.noaa.gov/maps/ncei/cdo/daily
result_dir: The directory to put the result
years: list with the begin and end year

Output
csv file with data
"""
import os
import wget
import pandas as pd


def get_data(stations_file, result_dir, years):
    table = pd.read_csv(stations_file, index_col=1,
                           usecols=['STATION', 'STATION_ID'])
    stations = table['STATION_ID'].unique().tolist()

    stations = [stations[i:i + 1] for i in range(0, len(stations), 1)]

    all_files = []

    start_year = str(years[0])
    end_year = str(years[1])

    for i, stations_batch in enumerate(stations):
        stations_str = ','.join(map(str, stations_batch))

        url = f"https://www.ncei.noaa.gov/access/services/data/v1" \
              f"?dataset=global-summary-of-the-day&dataTypes" \
              f"=LATITUDE,LONGITUDE,ELEVATION,NAME,TEMP,TEMP_ATTRIBUTES," \
              f"DEWP,DEWP_ATTRIBUTES,SLP,SLP_ATTRIBUTES,STP,STP_ATTRIBUTES," \
              f"VISIB,VISIB_ATTRIBUTES,WDSP,WDSP_ATTRIBUTES,MXSPD,GUST,MAX," \
              f"MAX_ATTRIBUTES,MIN,MIN_ATTRIBUTES,PRCP,PRCP_ATTRIBUTES,SNDP," \
              f"FRSHTT&stations={stations_str}&startDate={start_year}-01-01&endDate={end_year}-01-02"

        filename = f"data/data_{i}.csv"
        if os.path.exists(filename):
            os.remove(filename)
        wget.download(url, filename)

        all_files.append(filename)

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv(result_dir)

    for file_name in all_files:
        os.remove(file_name)

if __name__ == '__main__':
    st_file = "data/stations.csv"
    res_dir = "data/data.csv"
    dates = [2000, 2020]
    get_data(st_file, res_dir, dates)

    tb = pd.read_csv(res_dir, index_col=1, na_values='NA',
                     usecols=['STATION', 'DATE'])
    stations = tb['STATION'].unique().tolist()
    print(len(stations))
    print(stations)


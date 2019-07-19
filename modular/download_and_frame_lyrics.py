# code by Liam Haas-Neill
# edited by Eugene Klyshko

# First, install Lyric Genius from the github!

import lyricsgenius
client_token = 's86ExsILIhrEXrTfQmePOYXJ6jPT9KACHQ22dXs960suWEpwa4HQwUn56AB5Gsx7'
genius = lyricsgenius.Genius(client_token)

#don't need to run this again, already saved the lyrics
#artist = genius.search_artist("Lil Pump", sort="title")
#artist.save_lyrics()

import pandas as pd
import numpy as np
import json
import glob

#empty data frame

#get the song lyric names
lyric_files = glob.glob("*.json")
#
#load the lyrics into a dataframe
df = pd.DataFrame()
for i in range(len(lyric_files)):
    predf = pd.read_json(lyric_files[i],orient='index',typ='series')
    df = df.append(predf.songs)
    
#i dont think we care about the rest
data = df[['title','lyrics']]

data.to_csv('data/lyrics_titles_AutoPump.csv')
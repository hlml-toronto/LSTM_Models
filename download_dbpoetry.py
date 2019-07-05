"""

Getting all the poems from https://github.com/thundercomb/poetrydb/blob/master/README.md. I have to read them in and then save them to a txt file.

"""
import urllib.request, json, urllib.parse
import os


def create_txtfile_dbpoetry( data_file ):

    # Loads up list of titles available on poetrydb.org
    with urllib.request.urlopen( "http://poetrydb.org/title" ) as url:
        data = json.loads(url.read().decode())

    # write to a txt file
    with open('data/poems.txt', 'w') as f:
        for i in range( len( data['titles'] ) ): # go through titles
            url_titles = urllib.parse.quote( "http://poetrydb.org/title/" + data['titles'][i], safe=':/?*=\'') # make sure there ar eno spaces
            with urllib.request.urlopen( url_titles ) as url:
                poem = json.loads(url.read().decode())
            if 'status' not in poem: # problem with some poem titles not having a page, returning 'status': '404'
                for item in poem[0]['lines']: # write to file.
                    f.write("%s \n" % item)
                print(i)
                f.write(" \n \n \n \n ") # separate poems

import requests
import time
import os
import sys
from urllib.request import urlretrieve
from random import choice
from PIL import Image

url = "https://api.themoviedb.org/3/genre/movie/list"

payload = {'api_key': '7f2e30e73b023e51464893ee9ab03b1d',
           'language': 'en-US'}
response = requests.request("GET", url, data=payload).json()
time.sleep(0.25) # always sleep 1/4 of a second after any request, so that we only make 4 per second

# collect the genres we are interested in
genres = {'Action': 0,
          'Comedy': 0,
          'Documentary': 0,
          'Horror': 0,
          'Western': 0}
for genre in response['genres']:
    if genre['name'] in genres.keys():
        genres[genre['name']] = genre['id']

# First, find out how many pages of results there are for each genre
genre_pages = {}
for genre in genres:
    url = "https://api.themoviedb.org/3/discover/movie"
    payload = {'api_key': '7f2e30e73b023e51464893ee9ab03b1d',
               'language': 'en-US',
               'sort_by': 'popularity.desc',
               'include_adult': 'false',
               'include_video': 'true',
               'with_genres': str(genres[genre])}
    response = requests.request("GET", url, data=payload).json()
    time.sleep(0.3)
    genre_pages[genre] = response['total_pages']
# find the smallest genre
min_genre_pages = min(genre_pages.values())

# create directories for each genre
for genre_name in genres:
    if not os.path.exists('data_new/%s' % genre_name):
        os.makedirs('data_new/%s' % genre_name)

# download posters for each genre
downloaded_movies = set()
for genre in genres:
    print("Downloading posters for", genre)
    # to avoid imbalance, cap pages at 1.25x the min page count
    num_pages = min(genre_pages[genre], int(min_genre_pages*1.25))
    for page in range(1,num_pages+1):
        sys.stdout.write("Progress: Page %d of %d" % (page, num_pages))
        url = "https://api.themoviedb.org/3/discover/movie"
        payload = {'api_key': '7f2e30e73b023e51464893ee9ab03b1d',
                   'language': 'en-US',
                   'sort_by': 'popularity.desc',
                   'include_adult': 'false',
                   'include_video': 'true',
                   'page': str(page),
                   'with_genres': str(genres[genre])}
        response = requests.request("GET", url, data=payload).json()
        time.sleep(0.3)
        for result in response['results']:
            if 'poster_path' in result and result['poster_path'] not in downloaded_movies:
                # first, we need to determine the movie's genre. Most movies will have more than
                # one, so we will randomly pick one genre (to avoid having duplicates)
                if 'genre_ids' in result:
                    result_genres = [g_id for g_id in result['genre_ids'] if g_id in genres.values()]
                    result_genre = choice(result_genres)
                else:
                    result_genre = genres[genre]
                genre_name = next(key for key in genres if genres[key]==result_genre)
                try:
                    # download and save the poster
                    image_name = "data_new/%s/%s" % (genre_name, result['poster_path'])
                    urlretrieve("http://image.tmdb.org/t/p/w500/%s" % result['poster_path'], image_name)
                    # resize the image to a standard size for use with neural network batch training
                    im = Image.open(image_name)
                    im = im.resize((128,int(1.5*128)), Image.ANTIALIAS)
                    im.save(image_name)
                    # save this movie to our history so we don't redownload it under a second genre
                    downloaded_movies.add(result['poster_path'])
                except:
                    pass
        sys.stdout.write('\r')
    sys.stdout.write('\n')

from flask import Flask, render_template, request, redirect, url_for
import spotipy
from spotipy.oauth2 import SpotifyOAuth

test_v = "0.0.0"

app = Flask(__name__)

# Spotify API credentials
SPOTIPY_CLIENT_ID = 'your_spotify_client_id'
SPOTIPY_CLIENT_SECRET = 'your_spotify_client_secret'
SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:50505/callback'

sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                        client_secret=SPOTIPY_CLIENT_SECRET,
                        redirect_uri=SPOTIPY_REDIRECT_URI,
                        scope='user-library-read')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50505)

import sqlite3
import sys

dbname = "dbhunt.db"

connection = sqlite3.connect(dbname)
cursor = connection.cursor()

# ASSIGNMENT 1
# artists with Jazz tracks in the tracks table, order by artist name, a-z
# Output shows artists name and tracks name only
sql_statement_1 = """
SELECT artists.Name, tracks.Name
FROM tracks
JOIN albums ON tracks.AlbumId = albums.AlbumId
JOIN artists ON albums.ArtistId = artists.ArtistId
JOIN genres ON tracks.GenreId = genres.GenreId
WHERE genres.Name = 'Jazz'
ORDER BY artists.Name ASC
"""

# ASSIGNMENT 2
# the 5 artist with the most tracks in the tracks table of the Jazz genre, highest no. of tracks first.
# Output shows artists name and their no. of tracks only
sql_statement_2 = """
SELECT artists.Name, COUNT(tracks.TrackId) AS num_tracks
FROM tracks
JOIN albums ON tracks.AlbumId = albums.AlbumId
JOIN artists ON albums.ArtistId = artists.ArtistId
JOIN genres ON tracks.GenreId = genres.GenreId
WHERE genres.Name = 'Jazz'
GROUP BY artists.ArtistId
ORDER BY num_tracks DESC
LIMIT 5
"""

try:
	results = cursor.execute(sql_statement_1) # and then 2 of course
except Exception as e:
	sys.exit(e)
for r in results:
	print(r)

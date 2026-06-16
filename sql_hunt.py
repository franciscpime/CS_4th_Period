import sqlite3
import sys

dbname = "dbhunt.db"
connection = sqlite3.connect(dbname)
cursor = connection.cursor()

# =============================================================
# EXERCÍCIO 1 — fácil
# Lista todos os géneros existentes, ordenados alfabeticamente
# =============================================================
sql_statement_1 = """
SELECT Name
FROM genres
ORDER BY name ASC
"""

# =============================================================
# EXERCÍCIO 2 — fácil/médio
# Lista todos os artistas de Jazz com as suas músicas
# Ordenado por nome do artista (a-z), depois nome da música
# Mostrar apenas: nome do artista, nome da música
# =============================================================
sql_statement_2 = """
SELECT artists.Name, tracks.Name
FROM artists
JOIN albums ON artists.ArtistId = albums.ArtistId
JOIN tracks ON albums.AlbumId = tracks.AlbumId
JOIN genres ON tracks.GenreId = genres.GenreId
WHERE genres.Name = 'Jazz'
ORDER BY artists.Name ASC
"""

# =============================================================
# EXERCÍCIO 3 — médio
# Top 5 artistas de Jazz com mais músicas
# Ordenado pelo número de músicas (descendente)
# Mostrar apenas: nome do artista, número de músicas
# =============================================================
sql_statement_3 = """
SELECT artists.Name, COUNT(tracks.TrackId) AS music_num
FROM artists
JOIN albums ON artists.ArtistId = albums.ArtistId
JOIN tracks ON albums.AlbumId = tracks.AlbumId
JOIN genres ON tracks.GenreId = genres.GenreId
WHERE genres.Name = 'Jazz'
GROUP BY artists.ArtistId, artists.Name
ORDER BY music_num DESC
LIMIT 5
"""

# =============================================================
# Muda o número aqui para testar cada exercício (1, 2 ou 3)
# =============================================================
exercicio = 3

statements = {
    1: sql_statement_1,
    2: sql_statement_2,
    3: sql_statement_3,
}

try:
    results = cursor.execute(statements[exercicio])
except Exception as e:
    sys.exit(e)

for r in results:
    print(r)


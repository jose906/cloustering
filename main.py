import re
import json
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import umap
import hdbscan
import helper
import numpy as np




UMBRAL = 0.8


DB_CONFIG = {
    "host": "34.69.57.221",
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
}

TOPICS_ESPECIALES = [
    "__SOLO_LINK__",
    "__SALUDO__",
    "__PORTADA__",
    "__EPAPER__",
    "__PROMOCION__",
    "__PROGRAMACION__",
    "__RESUMEN__"
]

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)



def get_topic_embeddings():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT topic_id, embedding_vector
        FROM topic_embeddings
        ORDER BY topic_id
    """
    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    topics = []

    for row in results:
        topics.append({
            "topic_id": row["topic_id"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return topics


def get_tweet_embedding():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            t.tweetid,
            t.text,
            e.embedding_vector
        FROM Tweets t
        JOIN tweet_embeddings e
            ON t.tweetid = e.tweetid
        LEFT JOIN topic_tweets tt
            ON t.tweetid = tt.tweetid
        WHERE tt.tweetid IS NULL
          AND t.created >= NOW() - INTERVAL 14 DAY
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    tweets = []

    for row in results:
        tweets.append({
            "tweetid": row["tweetid"],
             "text": row["text"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return tweets

def get_especial_topics():

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

    query = f"""
        SELECT topic_id, topic_name
        FROM topics
        WHERE topic_name IN ({placeholders})
    """

    cursor.execute(query, TOPICS_ESPECIALES)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    topicos = {}

    for row in results:
        topicos[row["topic_name"]] = row["topic_id"]

    return topicos


def insert_tweets_topic(df):
    if df.empty:
        print("No hay tweets para insertar")
        return



    connection = get_db_connection()
    cursor = connection.cursor()

    insert_query = """
        INSERT INTO topic_tweets
            (topic_id, tweetid, similarity, assigned_at)
        VALUES
            (%s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            similarity = VALUES(similarity),
            assigned_at = NOW()
    """

    tweets_topic = df.to_dict(orient="records")
    data_to_insert = [
        (item["topic_id"], item["tweetid"], item["similarity"])
        for item in tweets_topic
    ]

    cursor.executemany(insert_query, data_to_insert)

    conteo_topics = Counter(item["topic_id"] for item in tweets_topic)

    update_query = """
        UPDATE topics
        SET
            total_tweets = total_tweets + %s,
            last_seen = NOW()
        WHERE topic_id = %s
    """

    data_update = [
        (cantidad, topic_id)
        for topic_id, cantidad in conteo_topics.items()
    ]

    cursor.executemany(update_query, data_update)

    connection.commit()

    cursor.close()
    connection.close()

    print("Tweets insertados:", len(tweets_topic))
    print("Topics actualizados:", len(conteo_topics))
    
def classify_tweets():
    df_topics = pd.DataFrame(get_topic_embeddings())
    df_tweets = pd.DataFrame(get_tweet_embedding())
    topicos_especiales = get_especial_topics()

    if df_topics.empty:
        print("No hay topics con embeddings")
        return

    if df_tweets.empty:
        print("No hay tweets nuevos para clasificar")
        return

    topic_matrix = np.array(df_topics["embedding"].tolist(), dtype=np.float32)
    tweet_matrix = np.array(df_tweets["embedding"].tolist(), dtype=np.float32)

    sim_matrix = cosine_similarity(tweet_matrix, topic_matrix)

    best_idx = np.argmax(sim_matrix, axis=1)
    best_score = np.max(sim_matrix, axis=1)

    asignados = []
    no_asignados = []

    for i in range(len(df_tweets)):
        tweetid = df_tweets.iloc[i]["tweetid"]
        texto = df_tweets.iloc[i]["text"]
        score = float(best_score[i])
        topic_especial = helper.detectar_topic_especial(texto)

        if topic_especial is not None:

            asignados.append({
                "tweetid": tweetid,
                "topic_id": topicos_especiales[topic_especial],
                "similarity": 1.0
            })
            print("--------------------------------")
            print("TÓPICO ESPECIAL DETECTADO")
            print("Tipo:", topic_especial)
            print("Tweet:", texto[:150])
            print("--------------------------------")


        if score >= UMBRAL:
            topic_id = int(df_topics.iloc[best_idx[i]]["topic_id"])


            asignados.append({
                "tweetid": tweetid,
                "topic_id": topic_id,
                "similarity": score
            })

        else:
            no_asignados.append({
            "tweetid": tweetid,
            "text": df_tweets.iloc[i]["text"],
            "embedding": df_tweets.iloc[i]["embedding"],
            "similarity": score
            })

    print("--------------------------------")
    print("Asignados:", len(asignados))
    print("No asignados:", len(no_asignados))
    print("--------------------------------")

    print(pd.DataFrame(asignados).head())
    print(pd.DataFrame(no_asignados).head())

    insert_tweets_topic(pd.DataFrame(asignados))

    return pd.DataFrame(no_asignados)

def detectar_nuevos_clusters(df_no_asignados):

    if len(df_no_asignados) < 20:
        print("Muy pocos tweets para detectar nuevos temas")
        return None

    embeddings = np.array(
        df_no_asignados["embedding"].tolist(),
        dtype=np.float32
    )

    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    embeddings_umap = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings_umap)

    df_clusters = df_no_asignados.copy()
    df_clusters["cluster"] = labels

    print(df_clusters["cluster"].value_counts())

    return df_clusters

def calcular_centroides_nuevos(df_clusters):

    if df_clusters is None or df_clusters.empty:
        print("No hay clusters para calcular centroides")
        return []

    columnas_requeridas = {"cluster", "embedding"}

    if not columnas_requeridas.issubset(df_clusters.columns):
        faltantes = columnas_requeridas - set(df_clusters.columns)
        print("Faltan columnas:", faltantes)
        return []

    # Excluir el ruido de HDBSCAN
    df_validos = df_clusters[
        df_clusters["cluster"] != -1
    ].copy()

    if df_validos.empty:
        print("No existen clusters válidos; todos son ruido")
        return []

    centroides_nuevos = []

    for cluster_id, grupo in df_validos.groupby("cluster"):

        matriz = np.array(
            grupo["embedding"].tolist(),
            dtype=np.float32
        )

        centroide = matriz.mean(axis=0)

        norma = np.linalg.norm(centroide)

        if norma == 0:
            print(
                f"Cluster {cluster_id} omitido: "
                "el centroide tiene norma cero"
            )
            continue

        centroide = centroide / norma

        centroides_nuevos.append({
            "cluster": int(cluster_id),
            "total_tweets": int(len(grupo)),
            "embedding": centroide.tolist()
        })

    print("Centroides nuevos calculados:", len(centroides_nuevos))

    return centroides_nuevos

def comparar_centroides_con_topics(
    centroides_nuevos,
    topics_existentes,
    umbral=0.80
):
    if not centroides_nuevos:
        print("No hay centroides nuevos")
        return pd.DataFrame()

    if not topics_existentes:
        print("No hay tópicos existentes")
        return pd.DataFrame()

    matriz_nuevos = np.array(
        [item["embedding"] for item in centroides_nuevos],
        dtype=np.float32
    )

    matriz_topics = np.array(
        [item["embedding"] for item in topics_existentes],
        dtype=np.float32
    )

    similitudes = cosine_similarity(
        matriz_nuevos,
        matriz_topics
    )

    resultados = []

    for i, centroide in enumerate(centroides_nuevos):

        mejor_indice = np.argmax(similitudes[i])
        mejor_similitud = float(similitudes[i][mejor_indice])

        topic_id = int(
            topics_existentes[mejor_indice]["topic_id"]
        )

        resultados.append({
            "cluster": centroide["cluster"],
            "total_tweets": centroide["total_tweets"],
            "topic_id_similar": topic_id,
            "similarity": mejor_similitud,
            "es_nuevo": mejor_similitud < umbral
        })

    return pd.DataFrame(resultados)

def preparar_clusters_similares(
    df_clusters,
    df_comparacion
):
    if df_clusters is None or df_clusters.empty:
        print("No hay clusters para procesar")
        return pd.DataFrame()

    if df_comparacion is None or df_comparacion.empty:
        print("No hay comparación de centroides")
        return pd.DataFrame()

    # Clusters que NO son nuevos
    df_similares = df_comparacion[
        df_comparacion["es_nuevo"] == False
    ].copy()

    if df_similares.empty:
        print("No hay clusters similares a tópicos existentes")
        return pd.DataFrame()

    asignaciones = []

    for _, comparacion in df_similares.iterrows():

        cluster_id = int(comparacion["cluster"])
        topic_id = int(comparacion["topic_id_similar"])
        similarity = float(comparacion["similarity"])

        tweets_cluster = df_clusters[
            df_clusters["cluster"] == cluster_id
        ]

        for _, tweet in tweets_cluster.iterrows():
            asignaciones.append({
                "tweetid": tweet["tweetid"],
                "topic_id": topic_id,
                "similarity": similarity
            })

    return pd.DataFrame(asignaciones)
def crear_topic_nuevo(cluster_id, centroide):

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        nombre_temporal = f"NUEVO_CLUSTER_{cluster_id}"

        insert_topic = """
            INSERT INTO topics (
                topic_name,
                topic_keywords,
                first_seen,
                last_seen,
                total_tweets,
                active,
                created_at
            )
            VALUES (
                %s,
                NULL,
                NOW(),
                NOW(),
                0,
                1,
                NOW()
            )
        """

        cursor.execute(
            insert_topic,
            (nombre_temporal,)
        )

        nuevo_topic_id = cursor.lastrowid

        insert_embedding = """
            INSERT INTO topic_embeddings (
                topic_id,
                embedding_vector,
                updated_at
            )
            VALUES (
                %s,
                %s,
                NOW()
            )
        """

        cursor.execute(
            insert_embedding,
            (
                nuevo_topic_id,
                json.dumps(centroide)
            )
        )

        connection.commit()

        print(
            f"Tópico creado: {nombre_temporal} "
            f"| topic_id: {nuevo_topic_id}"
        )

        return nuevo_topic_id

    except Exception as error:
        connection.rollback()
        print("Error creando el tópico:", error)
        return None

    finally:
        cursor.close()
        connection.close()

def crear_todos_los_topics_nuevos(df_topics_nuevos,centroides_nuevos,df_clusters):
    if df_topics_nuevos is None or df_topics_nuevos.empty:
        print("No hay tópicos nuevos para crear")
        return []

    resultados = []

    for _, fila in df_topics_nuevos.iterrows():

        cluster_id = int(fila["cluster"])

        centroide_item = next(
            (
                item
                for item in centroides_nuevos
                if int(item["cluster"]) == cluster_id
            ),
            None
        )

        if centroide_item is None:
            print(
                f"No se encontró el centroide "
                f"del cluster {cluster_id}"
            )
            continue

        nuevo_topic_id = crear_topic_nuevo(
            cluster_id=cluster_id,
            centroide=centroide_item["embedding"]
        )

        if nuevo_topic_id is None:
            print(
                f"No se pudo crear el tópico "
                f"del cluster {cluster_id}"
            )
            continue

        df_tweets_topic = preparar_tweets_nuevo_topic(
            df_clusters=df_clusters,
            cluster_id=cluster_id,
            nuevo_topic_id=nuevo_topic_id
        )

        if df_tweets_topic.empty:
            print(
                f"El cluster {cluster_id} "
                "no tiene tweets"
            )
            continue

        insert_tweets_topic(df_tweets_topic)

        resultados.append({
            "cluster": cluster_id,
            "topic_id": nuevo_topic_id,
            "total_tweets": len(df_tweets_topic)
        })

    print("--------------------------------")
    print("Nuevos tópicos creados:", len(resultados))
    print("--------------------------------")

    return resultados

def preparar_tweets_nuevo_topic(
    df_clusters,
    cluster_id,
    nuevo_topic_id
):
    grupo = df_clusters[
        df_clusters["cluster"] == cluster_id
    ].copy()

    if grupo.empty:
        print("No hay tweets para ese cluster")
        return pd.DataFrame()

    grupo["topic_id"] = nuevo_topic_id
    grupo["similarity"] = 1.0

    return grupo[
        ["tweetid", "topic_id", "similarity"]
    ]





if __name__ == "__main__":
    df_no_asignados = classify_tweets()
    df_clusters = detectar_nuevos_clusters(df_no_asignados)
    centroides_nuevos = calcular_centroides_nuevos(df_clusters)
    topics_existentes = get_topic_embeddings()

    df_comparacion = comparar_centroides_con_topics(centroides_nuevos,topics_existentes,umbral=UMBRAL)
    df_asignaciones_clusters = preparar_clusters_similares(df_clusters,df_comparacion)
    insert_tweets_topic(df_asignaciones_clusters)
    df_topics_nuevos = df_comparacion[df_comparacion["es_nuevo"] == True].copy()
    topics_creados = crear_todos_los_topics_nuevos( df_topics_nuevos=df_topics_nuevos,centroides_nuevos=centroides_nuevos,   df_clusters=df_clusters)
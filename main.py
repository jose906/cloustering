import os
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
from datetime import datetime


   

UMBRAL = 0.80
UMBRAL_NUEVO_TOPIC = 0.80
UMBRAL_TWEET_CLUSTER = 0.65


DB_CONFIG = {
            # IP pública o nombre interno de Cloud SQL
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "unix_socket": f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}",
    "charset": "utf8mb4",
    "port": "3306",
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

    if not os.environ.get("INSTANCE_CONNECTION_NAME"):
        raise RuntimeError(
            "Falta la variable de entorno INSTANCE_CONNECTION_NAME"
        )

    if not os.environ.get("DB_USER"):
        raise RuntimeError("Falta la variable de entorno DB_USER")

    if not os.environ.get("DB_PASS"):
        raise RuntimeError("Falta la variable de entorno DB_PASS")

    if not os.environ.get("DB_NAME"):
        raise RuntimeError("Falta la variable de entorno DB_NAME")

    return mysql.connector.connect(**DB_CONFIG)


def get_topic_embeddings():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

    query = f"""
        SELECT
            te.topic_id,
            te.embedding_vector
        FROM topic_embeddings te
        JOIN topics t
            ON t.topic_id = te.topic_id
        WHERE t.topic_name NOT IN ({placeholders})
        ORDER BY te.topic_id
    """

    cursor.execute(query, TOPICS_ESPECIALES)
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

    try:
        insert_query = """
            INSERT INTO topic_tweets
                (topic_id, tweetid, similarity, assigned_at)
            VALUES
                (%s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                tweetid = VALUES(tweetid)
        """

        tweets_topic = df.to_dict(orient="records")

        data_to_insert = [
            (item["topic_id"], item["tweetid"], item["similarity"])
            for item in tweets_topic
        ]

        cursor.executemany(insert_query, data_to_insert)

        conteo_topics = Counter(
            item["topic_id"]
            for item in tweets_topic
        )

        update_query = """
            UPDATE topics
            SET
                total_tweets = (
                    SELECT COUNT(*)
                    FROM topic_tweets
                    WHERE topic_tweets.topic_id = topics.topic_id
                ),
                last_seen = NOW()
            WHERE topic_id = %s
        """

        data_update = [
            (topic_id,)
            for topic_id in conteo_topics.keys()
        ]

        cursor.executemany(
            update_query,
            data_update
        )

        connection.commit()

        print("Tweets insertados:", len(tweets_topic))
        print("Topics actualizados:", len(conteo_topics))

    except Exception as error:

        connection.rollback()

        print(
            "Error insertando tweets en topics:",
            error
        )

        raise

    finally:
        cursor.close()
        connection.close()
    
def recalcular_centroide_topic(topic_id):

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT e.embedding_vector
        FROM topic_tweets tt
        JOIN tweet_embeddings e
            ON e.tweetid = tt.tweetid
        WHERE tt.topic_id = %s
    """

    cursor.execute(query, (topic_id,))
    resultados = cursor.fetchall()

    if not resultados:
        cursor.close()
        connection.close()
        print(f"Topic {topic_id}: no tiene embeddings")
        return

    embeddings = np.array(
        [
            json.loads(row["embedding_vector"])
            for row in resultados
        ],
        dtype=np.float32
    )

    centroide = embeddings.mean(axis=0)

    norma = np.linalg.norm(centroide)

    if norma == 0:
        cursor.close()
        connection.close()
        print(f"Topic {topic_id}: centroide con norma cero")
        return

    centroide = centroide / norma

    update_query = """
        UPDATE topic_embeddings
        SET
            embedding_vector = %s,
            updated_at = NOW()
        WHERE topic_id = %s
    """

    cursor.execute(
        update_query,
        (
            json.dumps(centroide.tolist()),
            topic_id
        )
    )

    connection.commit()

    cursor.close()
    connection.close()

    print(f"Centroide actualizado para topic {topic_id}")
    
def recalcular_total_tweets_topic(topic_id):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        query = """
            UPDATE topics
            SET total_tweets = (
                SELECT COUNT(*)
                FROM topic_tweets
                WHERE topic_tweets.topic_id = topics.topic_id
            )
            WHERE topic_id = %s
        """

        cursor.execute(query, (topic_id,))
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()
        
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

    BATCH_SIZE = 1000

    best_idx = []
    best_score = []

    for inicio in range(0, len(tweet_matrix), BATCH_SIZE):
        fin = inicio + BATCH_SIZE

        batch = tweet_matrix[inicio:fin]

        sim_batch = cosine_similarity(batch, topic_matrix)

        best_idx.extend(np.argmax(sim_batch, axis=1))
        best_score.extend(np.max(sim_batch, axis=1))

    best_idx = np.array(best_idx)
    best_score = np.array(best_score)
    asignados = []
    no_asignados = []

    for i in range(len(df_tweets)):
        tweetid = df_tweets.iloc[i]["tweetid"]
        texto = df_tweets.iloc[i]["text"]
        score = float(best_score[i])
        topic_especial = helper.detectar_topic_especial(texto)

        if topic_especial is not None:

            topic_especial_id = topicos_especiales.get(topic_especial)

            if topic_especial_id is not None:
                asignados.append({
                    "tweetid": tweetid,
                    "topic_id": topic_especial_id,
                    "similarity": 1.0
                })

                print("--------------------------------")
                print("TÓPICO ESPECIAL DETECTADO")
                print("Tipo:", topic_especial)
                print("Tweet:", texto[:150])
                print("--------------------------------")

                continue

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

    df_asignados = pd.DataFrame(asignados)

    if not df_asignados.empty:

        insert_tweets_topic(df_asignados)

        ids_especiales = set(topicos_especiales.values())

        topics_actualizados = (df_asignados[~df_asignados["topic_id"].isin(ids_especiales)]["topic_id"].drop_duplicates().tolist())

        for topic_id in topics_actualizados:
            recalcular_centroide_topic(int(topic_id))



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

def filtrar_clusters_por_calidad(df_clusters,similitud_minima=0.65, min_tweets=20):

    clusters_validos = []

    for cluster_id, grupo in df_clusters.groupby("cluster"):

        # -1 es ruido de HDBSCAN
        if cluster_id == -1:
            continue
        if len(grupo) < min_tweets:
            print(f"Cluster {cluster_id} descartado: "f"solo tiene {len(grupo)} tweets")
            continue

        embeddings = np.array(
            grupo["embedding"].tolist(),
            dtype=np.float32
        )

        centroide = embeddings.mean(axis=0)

        norma = np.linalg.norm(centroide)

        if norma == 0:
            continue

        centroide = centroide / norma

        similitudes = cosine_similarity(
            embeddings,
            centroide.reshape(1, -1)
        ).flatten()

        similitud_promedio = float(similitudes.mean())
        proporcion_fuertes = float(np.mean(similitudes >= 0.60))

        print(f"Cluster {cluster_id} | "f"Tweets: {len(grupo)} | "f"Similitud interna: {similitud_promedio:.3f} | "f"Tweets fuertes: {proporcion_fuertes:.2%}")

        if similitud_promedio >= similitud_minima and proporcion_fuertes >= 0.70:
            clusters_validos.append(cluster_id)

    return clusters_validos


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

def comparar_centroides_con_topics(centroides_nuevos,topics_existentes,umbral=0.80):
    
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

def preparar_clusters_similares(df_clusters,df_comparacion,topics_existentes):
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
        topic_item = next((item for item in topics_existentes if int(item["topic_id"]) == topic_id),None)

        if topic_item is None:
            continue

        topic_embedding = np.array(topic_item["embedding"],dtype=np.float32).reshape(1, -1)

        tweets_cluster = df_clusters[df_clusters["cluster"] == cluster_id]

        tweet_embeddings = np.array(tweets_cluster["embedding"].tolist(),dtype=np.float32)

        similitudes_tweets = cosine_similarity(tweet_embeddings,topic_embedding).flatten()
        
        print(
            f"Cluster {cluster_id} -> Topic {topic_id} | "
            f"Tweets: {len(similitudes_tweets)} | "
            f">=0.60: {np.sum(similitudes_tweets >= 0.60)} | "
            f">=0.65: {np.sum(similitudes_tweets >= 0.65)} | "
            f">=0.70: {np.sum(similitudes_tweets >= 0.70)} | "
            f">=0.75: {np.sum(similitudes_tweets >= 0.75)} | "
            f">=0.80: {np.sum(similitudes_tweets >= 0.80)} | "
            f"Promedio: {np.mean(similitudes_tweets):.3f} | "
            f"Máxima: {np.max(similitudes_tweets):.3f} | "
            f"Mínima: {np.min(similitudes_tweets):.3f}"
        )
        for j, (_, tweet) in enumerate(tweets_cluster.iterrows()):

            similitud_tweet = float(similitudes_tweets[j])

            if similitud_tweet < UMBRAL_TWEET_CLUSTER:
                continue

            asignaciones.append({
                "tweetid": tweet["tweetid"],
                "topic_id": topic_id,
                "similarity": similitud_tweet
            })
                
    return pd.DataFrame(asignaciones)
def crear_topic_nuevo(cluster_id, centroide):

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nombre_temporal = f"NUEVO_CLUSTER_{cluster_id}_{timestamp}"

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
        
        topics_actuales = get_topic_embeddings()

        comparacion_final = comparar_centroides_con_topics([centroide_item],topics_actuales,umbral=UMBRAL_NUEVO_TOPIC)

        if not comparacion_final.empty:
            if comparacion_final.iloc[0]["es_nuevo"] == False:

                topic_existente_id = int(
                    comparacion_final.iloc[0]["topic_id_similar"]
                )

                print(
                    f"Cluster {cluster_id} ya coincide con "
                    f"el topic {topic_existente_id}. "
                    "Se asignarán sus tweets al tópico existente."
                )

                df_cluster_existente = df_clusters[
                    df_clusters["cluster"] == cluster_id
                ].copy()

                topic_item = next(
                    (
                        item for item in topics_actuales
                        if int(item["topic_id"]) == topic_existente_id
                    ),
                    None
                )

                if topic_item is not None:

                    topic_embedding = np.array(
                        topic_item["embedding"],
                        dtype=np.float32
                    ).reshape(1, -1)

                    embeddings_cluster = np.array(
                        df_cluster_existente["embedding"].tolist(),
                        dtype=np.float32
                    )

                    similitudes = cosine_similarity(
                        embeddings_cluster,
                        topic_embedding
                    ).flatten()

                    asignaciones = []

                    for j, (_, tweet) in enumerate(
                        df_cluster_existente.iterrows()
                    ):

                        similitud = float(similitudes[j])

                        if similitud < UMBRAL_TWEET_CLUSTER:
                            continue

                        asignaciones.append({
                            "tweetid": tweet["tweetid"],
                            "topic_id": topic_existente_id,
                            "similarity": similitud
                        })

                    df_asignaciones = pd.DataFrame(asignaciones)

                    if not df_asignaciones.empty:
                        insert_tweets_topic(df_asignaciones)
                        recalcular_centroide_topic(topic_existente_id)

                continue

        nuevo_topic_id = crear_topic_nuevo(cluster_id=cluster_id,centroide=centroide_item["embedding"])

        if nuevo_topic_id is None:
            print(
                f"No se pudo crear el tópico "
                f"del cluster {cluster_id}"
            )
            continue

        df_tweets_topic = preparar_tweets_nuevo_topic(df_clusters=df_clusters,cluster_id=cluster_id,nuevo_topic_id=nuevo_topic_id,centroide=centroide_item["embedding"])

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

def preparar_tweets_nuevo_topic(df_clusters,cluster_id,nuevo_topic_id,centroide):
    grupo = df_clusters[
        df_clusters["cluster"] == cluster_id
    ].copy()

    if grupo.empty:
        print("No hay tweets para ese cluster")
        return pd.DataFrame()

    grupo["topic_id"] = nuevo_topic_id
    centroide_array = np.array(centroide,dtype=np.float32).reshape(1, -1)

    embeddings_tweets = np.array(grupo["embedding"].tolist(),dtype=np.float32)

    similitudes = cosine_similarity(embeddings_tweets,centroide_array).flatten()

    grupo["similarity"] = similitudes
    grupo["similarity"] = similitudes

    print(
        f"Nuevo topic cluster {cluster_id} | "
        f"Tweets originales: {len(grupo)} | "
        f">=0.60: {np.sum(grupo['similarity'] >= 0.60)} | "
        f"Promedio: {grupo['similarity'].mean():.3f} | "
        f"Mínima: {grupo['similarity'].min():.3f}"
    )


    grupo = grupo[
    grupo["similarity"] >= 0.60
    ].copy()

    return grupo[
        ["tweetid", "topic_id", "similarity"]
    ]



def ejecutar_procesamiento():

    print("========================================")
    print("INICIANDO PROCESAMIENTO DE NETVORA")
    print("========================================")

    df_no_asignados = classify_tweets()

    if df_no_asignados is None or df_no_asignados.empty:
        print("No hay tweets sin asignar.")
        return

    df_clusters = detectar_nuevos_clusters(df_no_asignados)

    if df_clusters is None or df_clusters.empty:
        print("No se pudieron detectar clusters.")
        return

    clusters_validos = filtrar_clusters_por_calidad(df_clusters,similitud_minima=0.65, min_tweets=20)

    if len(clusters_validos) == 0:
        print("No se encontraron clusters con calidad suficiente")
        return

    df_clusters = df_clusters[
        df_clusters["cluster"].isin(clusters_validos)
    ].copy()

    centroides_nuevos = calcular_centroides_nuevos(df_clusters)

    topics_existentes = get_topic_embeddings()

    df_comparacion = comparar_centroides_con_topics(centroides_nuevos,topics_existentes,umbral=UMBRAL_NUEVO_TOPIC)
    print("--------------------------------")
    print("COMPARACIÓN CLUSTERS VS TOPICS")
    print(df_comparacion.to_string(index=False))
    print("--------------------------------")

    df_asignaciones_clusters = preparar_clusters_similares(df_clusters,df_comparacion,topics_existentes)

    if not df_asignaciones_clusters.empty:
        insert_tweets_topic(df_asignaciones_clusters)
        
        topics_actualizados = (df_asignaciones_clusters["topic_id"].drop_duplicates().tolist())
    
        for topic_id in topics_actualizados:
            recalcular_centroide_topic(int(topic_id))


    df_topics_nuevos = df_comparacion[df_comparacion["es_nuevo"] == True].copy()

    topics_creados = crear_todos_los_topics_nuevos(df_topics_nuevos=df_topics_nuevos,centroides_nuevos=centroides_nuevos,df_clusters=df_clusters)

    print("========================================")
    print("PROCESAMIENTO TERMINADO")
    print("Topics creados:", len(topics_creados))
    print("========================================")


if __name__ == "__main__":
    ejecutar_procesamiento()



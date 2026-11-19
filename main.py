import os
import json
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import helper
from datetime import datetime


   

UMBRAL = 0.80
UMBRAL_NUEVO_TOPIC = 0.80
# Valores iniciales para calibrar con eventos etiquetados.
UMBRAL_TWEET_CLUSTER = 0.75
MARGEN_TOPIC = 0.03
APOYO_MINIMO_FUSION = 0.70
MIN_TWEETS_CLUSTER = 20
SIMILITUD_MINIMA_MIEMBRO = 0.65


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
        ORDER BY t.created, t.tweetid
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


def guardar_asignaciones(cursor, asignaciones):
    cursor.executemany(
        """
        INSERT INTO topic_tweets (topic_id, tweetid, similarity, assigned_at)
        VALUES (%s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE tweetid = VALUES(tweetid)
        """,
        [(fila.topic_id, fila.tweetid, fila.similarity)
         for fila in asignaciones.itertuples(index=False)],
    )
    cursor.executemany(
        """
        UPDATE topics
        SET total_tweets = (
            SELECT COUNT(*) FROM topic_tweets
            WHERE topic_tweets.topic_id = topics.topic_id
        ),
        last_seen = NOW()
        WHERE topic_id = %s
        """,
        [(topic_id,) for topic_id in asignaciones["topic_id"].drop_duplicates()],
    )


def insert_tweets_topic(df):
    if df.empty:
        return

    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        guardar_asignaciones(cursor, df)
        connection.commit()
    except Exception:
        connection.rollback()
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

    centroide = normalizar_vectores(embeddings).mean(axis=0)

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
        
def normalizar_vectores(vectores):
    matriz = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return np.divide(matriz, normas, out=np.zeros_like(matriz), where=normas > 0)


def mejores_topics(embeddings, topics):
    matriz_topics = np.asarray([topic["embedding"] for topic in topics], dtype=np.float32)
    for inicio in range(0, len(embeddings), 1000):
        lote = np.asarray(embeddings[inicio:inicio + 1000], dtype=np.float32)
        similitudes = cosine_similarity(lote, matriz_topics)
        indices = np.argmax(similitudes, axis=1)
        mejores = similitudes[np.arange(len(lote)), indices]
        if len(topics) > 1:
            segundos = np.partition(similitudes, -2, axis=1)[:, -2]
            margenes = mejores - segundos
        else:
            margenes = np.full(len(lote), np.inf)
        for indice, score, margen in zip(indices, mejores, margenes):
            yield int(topics[indice]["topic_id"]), float(score), float(margen)


def clasificar_lote(tweets, topics, topicos_especiales):
    asignados = []
    pendientes = []
    for tweet in tweets:
        especial = helper.detectar_topic_especial(tweet["text"])
        topic_id = topicos_especiales.get(especial)
        if topic_id is None:
            pendientes.append(tweet)
        else:
            asignados.append({"tweetid": tweet["tweetid"], "topic_id": topic_id, "similarity": 1.0})

    if not topics:
        return pd.DataFrame(asignados), pd.DataFrame(pendientes)

    no_asignados = []
    candidatos = mejores_topics([tweet["embedding"] for tweet in pendientes], topics)
    for tweet, (topic_id, score, margen) in zip(pendientes, candidatos):
        if score >= UMBRAL and margen >= MARGEN_TOPIC:
            asignados.append({"tweetid": tweet["tweetid"], "topic_id": topic_id, "similarity": score})
        else:
            no_asignados.append({**tweet, "similarity": score})

    return pd.DataFrame(asignados), pd.DataFrame(no_asignados)


def classify_tweets():
    tweets = get_tweet_embedding()
    if not tweets:
        return pd.DataFrame()

    especiales = get_especial_topics()
    asignados, pendientes = clasificar_lote(tweets, get_topic_embeddings(), especiales)
    print(f"Tweets asignados: {len(asignados)} | pendientes: {len(pendientes)}")
    if not asignados.empty:
        insert_tweets_topic(asignados)
        ids_especiales = set(especiales.values())
        for topic_id in asignados["topic_id"].drop_duplicates():
            if topic_id not in ids_especiales:
                recalcular_centroide_topic(int(topic_id))
    return pendientes


def detectar_nuevos_clusters(df_no_asignados):

    if len(df_no_asignados) < MIN_TWEETS_CLUSTER:
        print("Muy pocos tweets para detectar nuevos temas")
        return None

    embeddings = np.array(
        df_no_asignados["embedding"].tolist(),
        dtype=np.float32
    )

    reducer = umap.UMAP(
        n_neighbors=min(30, len(embeddings) - 1),
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    embeddings_umap = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TWEETS_CLUSTER,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings_umap)

    df_clusters = df_no_asignados.copy()
    df_clusters["cluster"] = labels

    print(df_clusters["cluster"].value_counts())

    return df_clusters

def filtrar_clusters_por_calidad(
    df_clusters, similitud_minima=SIMILITUD_MINIMA_MIEMBRO, min_tweets=MIN_TWEETS_CLUSTER
):
    grupos_validos = []
    for cluster_id, grupo in df_clusters.groupby("cluster"):
        if cluster_id == -1:
            continue

        grupo = grupo.drop_duplicates(subset="tweetid")
        if len(grupo) < min_tweets:
            continue
        vectores = normalizar_vectores(grupo["embedding"].tolist())
        while len(grupo) >= min_tweets:
            # Excluir el propio tweet del centro con el que se compara.
            otros = vectores.sum(axis=0) - vectores
            similitudes = np.einsum("ij,ij->i", vectores, normalizar_vectores(otros))
            fuertes = similitudes >= similitud_minima
            if fuertes.all():
                grupos_validos.append(grupo)
                break
            grupo = grupo.loc[fuertes]
            vectores = vectores[fuertes]
        print(f"Cluster {cluster_id}: {len(grupo)} miembros tras el filtro")

    if grupos_validos:
        return pd.concat(grupos_validos)
    return df_clusters.iloc[:0].copy()


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

        centroide = normalizar_vectores(matriz).mean(axis=0)

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
    centroides_nuevos, topics_existentes, umbral=UMBRAL_NUEVO_TOPIC
):
    if not centroides_nuevos:
        print("No hay centroides nuevos")
        return pd.DataFrame()

    embeddings = [item["embedding"] for item in centroides_nuevos]
    candidatos = (
        mejores_topics(embeddings, topics_existentes)
        if topics_existentes
        else [(None, None, None)] * len(centroides_nuevos)
    )
    resultados = []
    for centroide, (topic_id, score, margen) in zip(centroides_nuevos, candidatos):
        resultados.append({
            "cluster": centroide["cluster"],
            "total_tweets": centroide["total_tweets"],
            "topic_id_similar": topic_id,
            "similarity": score,
            "es_nuevo": score is None or score < umbral,
            "fusion_ambigua": margen is not None and margen < MARGEN_TOPIC,
        })

    return pd.DataFrame(resultados)

def preparar_clusters_similares(df_clusters, df_comparacion, topics_existentes):
    if df_comparacion.empty or not topics_existentes:
        return pd.DataFrame()

    asignaciones = []
    for comparacion in df_comparacion.itertuples(index=False):
        if comparacion.es_nuevo:
            continue
        if comparacion.fusion_ambigua:
            print(f"Cluster {comparacion.cluster}: fusión ambigua")
            continue
        grupo = df_clusters[df_clusters["cluster"] == comparacion.cluster]
        if grupo.empty:
            continue
        candidatos = mejores_topics(grupo["embedding"].tolist(), topics_existentes)
        aceptados = []
        for tweetid, (topic_id, score, margen) in zip(grupo["tweetid"], candidatos):
            if (topic_id == comparacion.topic_id_similar
                    and score >= UMBRAL_TWEET_CLUSTER and margen >= MARGEN_TOPIC):
                aceptados.append({"tweetid": tweetid, "topic_id": topic_id, "similarity": score})

        if len(aceptados) / len(grupo) >= APOYO_MINIMO_FUSION:
            asignaciones.extend(aceptados)
        else:
            print(f"Cluster {comparacion.cluster}: apoyo insuficiente ({len(aceptados)}/{len(grupo)})")

    return pd.DataFrame(asignaciones)


def crear_topic_nuevo(cluster_id, centroide, asignaciones):

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

        guardar_asignaciones(cursor, asignaciones.assign(topic_id=nuevo_topic_id))
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

def crear_todos_los_topics_nuevos(df_topics_nuevos, centroides_nuevos, df_clusters):
    centroides_por_cluster = {item["cluster"]: item for item in centroides_nuevos}
    resultados = []
    for fila in df_topics_nuevos.itertuples(index=False):
        cluster_id = fila.cluster
        centroide = centroides_por_cluster[cluster_id]
        grupo = df_clusters[df_clusters["cluster"] == cluster_id]
        if len(grupo) < MIN_TWEETS_CLUSTER:
            continue

        # El catálogo puede cambiar al crear o fusionar otros grupos del lote.
        topics_actuales = get_topic_embeddings()
        comparacion = comparar_centroides_con_topics(
            [centroide], topics_actuales, umbral=UMBRAL_NUEVO_TOPIC
        )
        if not comparacion.iloc[0]["es_nuevo"]:
            asignaciones = preparar_clusters_similares(grupo, comparacion, topics_actuales)
            if not asignaciones.empty:
                insert_tweets_topic(asignaciones)
                recalcular_centroide_topic(int(asignaciones.iloc[0]["topic_id"]))
            continue

        similitudes = cosine_similarity(
            np.asarray(grupo["embedding"].tolist(), dtype=np.float32),
            np.asarray([centroide["embedding"]], dtype=np.float32),
        ).ravel()
        asignaciones = grupo[["tweetid"]].assign(similarity=similitudes)
        topic_id = crear_topic_nuevo(cluster_id, centroide["embedding"], asignaciones)
        if topic_id is not None:
            resultados.append({
                "cluster": cluster_id, "topic_id": topic_id, "total_tweets": len(grupo)
            })

    return resultados


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

    df_clusters = filtrar_clusters_por_calidad(df_clusters)
    if df_clusters.empty:
        print("No se encontraron clusters con calidad suficiente")
        return

    centroides_nuevos = calcular_centroides_nuevos(df_clusters)

    if not centroides_nuevos:
        return

    topics_existentes = get_topic_embeddings()

    df_comparacion = comparar_centroides_con_topics(centroides_nuevos, topics_existentes)
    print("--------------------------------")
    print("COMPARACIÓN CLUSTERS VS TOPICS")
    print(df_comparacion.to_string(index=False))
    print("--------------------------------")

    df_asignaciones_clusters = preparar_clusters_similares(df_clusters, df_comparacion, topics_existentes)

    if not df_asignaciones_clusters.empty:
        insert_tweets_topic(df_asignaciones_clusters)
        
        topics_actualizados = (df_asignaciones_clusters["topic_id"].drop_duplicates().tolist())
    
        for topic_id in topics_actualizados:
            recalcular_centroide_topic(int(topic_id))


    df_topics_nuevos = df_comparacion[df_comparacion["es_nuevo"]].copy()

    topics_creados = crear_todos_los_topics_nuevos(df_topics_nuevos, centroides_nuevos, df_clusters)

    print("========================================")
    print("PROCESAMIENTO TERMINADO")
    print("Topics creados:", len(topics_creados))
    print("========================================")


if __name__ == "__main__":
    ejecutar_procesamiento()



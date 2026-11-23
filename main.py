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
import re



   

UMBRAL = 0.80
UMBRAL_NUEVO_TOPIC = 0.80
# Valores iniciales para calibrar con eventos etiquetados.
UMBRAL_TWEET_CLUSTER = 0.75
MARGEN_TOPIC = 0.03
APOYO_MINIMO_FUSION = 0.70

MIN_TWEETS_CLUSTER = 40
MIN_TEXTOS_UNICOS_CLUSTER = 30
SIMILITUD_MINIMA_MIEMBRO = 0.70

REFERENCIAS_POR_TOPIC = 150
CANDIDATOS_REFERENCIAS = 600

MIN_REFERENCIAS_TOPIC = 10

UMBRAL_REFERENCIA = 0.82
MIN_COINCIDENCIAS_REFERENCIA = 3
UMBRAL_PROMEDIO_TOP5 = 0.83

SIMILITUD_MAX_ENTRE_REFERENCIAS = 0.98

UMBRAL_CANDIDATO = 0.65
CANDIDATOS_POR_TWEET = 2
MARGEN_REFERENCIAS = 0.03

# Criterios conservadores de creación
UMBRAL_NOVEDAD_CENTROIDE = 0.78
MAX_FRACCION_COINCIDENCIAS_EXISTENTES = 0.10


# Estabilidad de los centroides existentes
PESO_CENTROIDE_NUEVO = 0.20
SIMILITUD_MINIMA_ACTUALIZACION = 0.97
MIN_EMBEDDINGS_RECALCULO = 10

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


def recalcular_centroide_topic(topic_id, simular=False):

    topic_id = int(topic_id)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:

        # 1. Obtener centroide actual
        cursor.execute(
            """
            SELECT embedding_vector
            FROM topic_embeddings
            WHERE topic_id = %s
            """,
            (topic_id,)
        )

        actual = cursor.fetchone()

        if not actual:
            print(f"Topic {topic_id}: sin centroide")
            return False

        centroide_anterior = np.asarray(
            json.loads(actual["embedding_vector"]),
            dtype=np.float64
        )

        norma_anterior = np.linalg.norm(
            centroide_anterior
        )

        if (
            not np.isfinite(centroide_anterior).all()
            or norma_anterior <= 0
        ):
            return False

        centroide_anterior /= norma_anterior

        # 2. Obtener los embeddings existentes
        cursor.execute(
            """
            SELECT
                tt.tweetid,
                t.text,
                e.embedding_vector
            FROM topic_tweets tt
            JOIN Tweets t
                ON t.tweetid = tt.tweetid
            JOIN tweet_embeddings e
                ON e.tweetid = tt.tweetid
            WHERE tt.topic_id = %s
            """,
            (topic_id,)
        )

        resultados = cursor.fetchall()

        if not resultados:
            return False

        df = pd.DataFrame([
            {
                "tweetid": row["tweetid"],
                "text": row["text"],
                "embedding": json.loads(
                    row["embedding_vector"]
                )
            }
            for row in resultados
        ])

        # 3. Calcular centroide robusto
        nuevo, utilizados, mediana = (
            calcular_centroide_robusto(df)
        )

        if (
            nuevo is None
            or utilizados < MIN_EMBEDDINGS_RECALCULO
        ):
            print(
                f"Topic {topic_id}: "
                "embeddings insuficientes"
            )
            return False

        centroide_nuevo = np.asarray(
            nuevo,
            dtype=np.float64
        )

        if centroide_nuevo.shape != centroide_anterior.shape:
            print(f"Topic {topic_id}: dimensiones incompatibles")
            return False

        # 4. Medir cuánto cambió el centroide
        similitud = float(
            centroide_anterior @ centroide_nuevo
        )

        print(f"\nTOPIC {topic_id}")
        print(f"Tweets: {len(df)}")
        print(f"Embeddings únicos: {utilizados}")
        print(f"Similitud mediana: {mediana:.4f}")
        print(f"Similitud antiguo/nuevo: {similitud:.4f}")

        # 5. Evitar cambios demasiado grandes
        if similitud < SIMILITUD_MINIMA_ACTUALIZACION:

            print(
                f"Topic {topic_id}: actualización bloqueada "
                "por cambio excesivo"
            )

            return False

        # 6. Combinar centroide anterior y nuevo
        centroide_final = (
            (1 - PESO_CENTROIDE_NUEVO)
            * centroide_anterior
            +
            PESO_CENTROIDE_NUEVO
            * centroide_nuevo
        )

        norma = np.linalg.norm(centroide_final)

        if norma <= 0:
            return False

        centroide_final /= norma

        if simular:
            print("SIMULACIÓN: centroide no guardado")
            return True

        # 7. Guardar en la misma tabla
        cursor.execute(
            """
            UPDATE topic_embeddings
            SET embedding_vector = %s,
                updated_at = NOW()
            WHERE topic_id = %s
            """,
            (
                json.dumps(centroide_final.tolist()),
                topic_id
            )
        )

        connection.commit()

        print(f"Topic {topic_id}: centroide actualizado")

        return True

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()
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

        especial = helper.detectar_topic_especial(
            tweet["text"]
        )

        topic_id = topicos_especiales.get(especial)

        if topic_id is not None:

            asignados.append({
                "tweetid": tweet["tweetid"],
                "topic_id": int(topic_id),
                "similarity": 1.0
            })

        else:
            pendientes.append(tweet)

    if not topics or not pendientes:
        return (
            pd.DataFrame(
                asignados,
                columns=["tweetid", "topic_id", "similarity"]
            ),
            pd.DataFrame(pendientes)
        )

    candidatos = obtener_candidatos_topics(
        [t["embedding"] for t in pendientes],
        topics
    )

    ids_topics = {
        topic_id
        for opciones in candidatos
        for topic_id, _ in opciones
    }

    referencias = get_referencias_topics(
        ids_topics,
        topics
    )

    no_asignados = []

    for tweet, opciones in zip(pendientes, candidatos):

        resultado = seleccionar_topic_preciso(
            tweet["embedding"],
            opciones,
            referencias
        )

        if resultado is None:

            no_asignados.append(tweet)
            continue

        asignados.append({
            "tweetid": tweet["tweetid"],
            "topic_id": resultado["topic_id"],
            "similarity": resultado["similarity"]
        })

    print(
        f"Clasificación precisa: "
        f"{len(asignados)} asignados | "
        f"{len(no_asignados)} pendientes"
    )

    return (
        pd.DataFrame(
            asignados,
            columns=["tweetid", "topic_id", "similarity"]
        ),
        pd.DataFrame(no_asignados)
    )
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
    df_clusters,
    similitud_minima=SIMILITUD_MINIMA_MIEMBRO,
    min_tweets=MIN_TWEETS_CLUSTER
):
    grupos_validos = []

    for cluster_id, grupo in df_clusters.groupby("cluster"):

        if cluster_id == -1:
            continue

        grupo = grupo.drop_duplicates(subset="tweetid").copy()

        if len(grupo) < min_tweets:
            continue

        vectores = normalizar_vectores(
            grupo["embedding"].tolist()
        )

        while len(grupo) >= min_tweets:

            # Comparación con el resto del grupo,
            # excluyendo el propio tweet.
            otros = vectores.sum(axis=0) - vectores

            similitudes = np.einsum(
                "ij,ij->i",
                vectores,
                normalizar_vectores(otros)
            )

            fuertes = similitudes >= similitud_minima

            if fuertes.all():

                textos = (
                    grupo["text"]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

                textos_unicos = textos[
                    textos != ""
                ].nunique()

                if textos_unicos < MIN_TEXTOS_UNICOS_CLUSTER:

                    print(
                        f"Cluster {cluster_id} rechazado: "
                        f"{textos_unicos} textos únicos "
                        f"de {len(grupo)} tweets"
                    )

                    break

                grupos_validos.append(grupo)

                print(
                    f"Cluster {cluster_id} válido: "
                    f"{len(grupo)} tweets | "
                    f"{textos_unicos} textos únicos"
                )

                break

            grupo = grupo.loc[fuertes].copy()
            vectores = vectores[fuertes]

        print(
            f"Cluster {cluster_id}: "
            f"{len(grupo)} miembros tras el filtro"
        )

    if grupos_validos:
        return pd.concat(grupos_validos)

    return df_clusters.iloc[:0].copy()





def calcular_centroides_nuevos(df_clusters):

    if df_clusters is None or df_clusters.empty:
        print("No hay clusters para calcular centroides")
        return []

    df_validos = df_clusters[
        df_clusters["cluster"] != -1
    ].copy()

    centroides_nuevos = []

    for cluster_id, grupo in df_validos.groupby("cluster"):

        centroide, utilizados, similitud_mediana = (
            calcular_centroide_robusto(grupo)
        )

        if centroide is None:
            print(f"Cluster {cluster_id}: centroide inválido")
            continue

        if utilizados < MIN_TEXTOS_UNICOS_CLUSTER:
            print(
                f"Cluster {cluster_id}: "
                f"muestra insuficiente ({utilizados})"
            )
            continue

        centroides_nuevos.append({
            "cluster": int(cluster_id),
            "total_tweets": int(len(grupo)),
            "embedding": centroide
        })

        print(
            f"Cluster {cluster_id} | "
            f"Tweets: {len(grupo)} | "
            f"Embeddings únicos: {utilizados} | "
            f"Similitud mediana: {similitud_mediana:.4f}"
        )

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

def preparar_clusters_similares(
    df_clusters,
    df_comparacion,
    topics_existentes
):

    columnas = ["tweetid", "topic_id", "similarity"]

    if df_comparacion.empty or not topics_existentes:
        return pd.DataFrame(columns=columnas)

    grupos = []

    for fila in df_comparacion.itertuples(index=False):

        if fila.es_nuevo or fila.fusion_ambigua:
            continue

        if pd.isna(fila.topic_id_similar):
            continue

        grupo = df_clusters[
            df_clusters["cluster"] == fila.cluster
        ].copy()

        if grupo.empty:
            continue

        grupos.append((
            fila.cluster,
            int(fila.topic_id_similar),
            grupo
        ))

    if not grupos:
        return pd.DataFrame(columns=columnas)

    candidatos_grupos = []
    ids_topics = set()

    for cluster_id, destino, grupo in grupos:

        candidatos = obtener_candidatos_topics(
            grupo["embedding"].tolist(),
            topics_existentes
        )

        candidatos_grupos.append(candidatos)

        ids_topics.update(
            topic_id
            for opciones in candidatos
            for topic_id, _ in opciones
        )

    referencias = get_referencias_topics(
        ids_topics,
        topics_existentes
    )

    asignaciones = []

    for (cluster_id, destino, grupo), candidatos in zip(
        grupos,
        candidatos_grupos
    ):

        aceptados = []

        for tweet, opciones in zip(
            grupo.itertuples(index=False),
            candidatos
        ):

            resultado = seleccionar_topic_preciso(
                tweet.embedding,
                opciones,
                referencias
            )

            if resultado is None:
                continue

            if resultado["topic_id"] != destino:
                continue

            aceptados.append({
                "tweetid": tweet.tweetid,
                "topic_id": destino,
                "similarity": resultado["similarity"]
            })

        apoyo = len(aceptados) / len(grupo)

        if apoyo >= APOYO_MINIMO_FUSION:
            asignaciones.extend(aceptados)
        else:
            print(
                f"Cluster {cluster_id}: "
                f"fusión rechazada "
                f"({len(aceptados)}/{len(grupo)})"
            )

    return pd.DataFrame(
        asignaciones,
        columns=columnas
    )

def diagnosticar_fusiones(df_clusters, df_comparacion, topics):

    filas = [
        fila
        for fila in df_comparacion.itertuples(index=False)
        if pd.notna(fila.topic_id_similar)
    ]

    referencias = get_referencias_topics(
        [int(f.topic_id_similar) for f in filas],
        topics
    )

    for fila in filas:

        topic_id = int(fila.topic_id_similar)

        grupo = df_clusters[
            df_clusters["cluster"] == fila.cluster
        ]

        refs = referencias.get(topic_id)

        candidatos = list(
            mejores_topics(
                grupo["embedding"].tolist(),
                topics
            )
        )

        mismo_topic = 0
        pasa_umbral = 0
        pasa_margen = 0
        pasa_coincidencias = 0
        pasa_promedio = 0
        pasa_ambos = 0

        maximos = []
        promedios = []

        for tweet, (mejor_id, score, margen) in zip(
            grupo.itertuples(index=False),
            candidatos
        ):

            if mejor_id != topic_id:
                continue

            mismo_topic += 1

            if score < UMBRAL:
                continue

            pasa_umbral += 1

            if margen < MARGEN_TOPIC:
                continue

            pasa_margen += 1

            if refs is None or len(refs) < MIN_REFERENCIAS_TOPIC:
                continue

            vector = normalizar_vectores(
                [tweet.embedding]
            )[0]

            similitudes = refs @ vector

            top5 = np.sort(similitudes)[-5:]

            maximos.append(float(top5[-1]))
            promedios.append(float(np.mean(top5)))

            coincidencias = int(
                np.sum(similitudes >= UMBRAL_REFERENCIA)
            )

            cumple_coincidencias = (
                coincidencias >= MIN_COINCIDENCIAS_REFERENCIA
            )

            cumple_promedio = (
                np.mean(top5) >= UMBRAL_PROMEDIO_TOP5
            )

            if cumple_coincidencias:
                pasa_coincidencias += 1

            if cumple_promedio:
                pasa_promedio += 1

            if cumple_coincidencias and cumple_promedio:
                pasa_ambos += 1

        print("\n" + "=" * 60)
        print(f"CLUSTER {fila.cluster} -> TOPIC {topic_id}")
        print("=" * 60)

        print("Tweets del cluster:", len(grupo))
        print("Referencias disponibles:",
              0 if refs is None else len(refs))

        print("Mejor tópico correcto:", mismo_topic)
        print("Superan umbral centroide:", pasa_umbral)
        print("Superan margen:", pasa_margen)
        print("Superan coincidencias:", pasa_coincidencias)
        print("Superan promedio top5:", pasa_promedio)
        print("Superan ambas condiciones:", pasa_ambos)

        if maximos:
            print(
                "Mediana mejor referencia:",
                round(float(np.median(maximos)), 4)
            )

            print(
                "Mediana promedio top5:",
                round(float(np.median(promedios)), 4)
            )
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

def crear_todos_los_topics_nuevos(
    df_topics_nuevos,
    centroides_nuevos,
    df_clusters,
    df_fusiones=None,
    simular=False
):

    centroides_por_cluster = {
        item["cluster"]: item
        for item in centroides_nuevos
    }

    resultados = []

    ids_fusionados = set()

    if df_fusiones is not None and not df_fusiones.empty:
        ids_fusionados = set(
            df_fusiones["tweetid"]
        )

    for fila in df_topics_nuevos.itertuples(index=False):

        cluster_id = int(fila.cluster)

        centroide = centroides_por_cluster.get(
            cluster_id
        )

        if centroide is None:
            continue

        grupo = df_clusters[
            df_clusters["cluster"] == cluster_id
        ].copy()

        if len(grupo) < MIN_TWEETS_CLUSTER:
            continue

        # Evitar crear un tópico si este cluster
        # ya consiguió una fusión aceptada.
        if any(
            tweetid in ids_fusionados
            for tweetid in grupo["tweetid"]
        ):
            print(
                f"Cluster {cluster_id}: "
                "ya tiene una fusión aceptada"
            )
            continue

        # Obtener catálogo actualizado.
        topics_actuales = get_topic_embeddings()

        comparacion = comparar_centroides_con_topics(
            [centroide],
            topics_actuales
        )

        if comparacion.empty:
            continue

        fila_comparacion = comparacion.iloc[0]

        score = fila_comparacion["similarity"]

        if pd.notna(score):
            score = float(score)
        else:
            score = None

        # Si se parece demasiado a un tópico
        # existente, no crear automáticamente.
        if (
            score is not None
            and score >= UMBRAL_NOVEDAD_CENTROIDE
        ):
            print(
                f"Cluster {cluster_id}: PENDIENTE | "
                f"Similitud con tópico existente: {score:.4f}"
            )
            continue

        # Evaluación adicional con embeddings
        # de las publicaciones ya existentes.
        puede_crearse, fraccion = evaluar_novedad_cluster(
            grupo,
            topics_actuales
        )

        if not puede_crearse:
            print(
                f"Cluster {cluster_id}: PENDIENTE | "
                f"Coincidencias existentes: {fraccion}"
            )
            continue

        # Calcular similitudes con el centroide
        # robusto que ya construimos.
        similitudes = cosine_similarity(
            np.asarray(
                grupo["embedding"].tolist(),
                dtype=np.float32
            ),
            np.asarray(
                [centroide["embedding"]],
                dtype=np.float32
            )
        ).ravel()

        asignaciones = grupo[["tweetid"]].assign(
            similarity=similitudes
        )

        if simular:
            print(
                f"Cluster {cluster_id}: "
                f"CANDIDATO A NUEVO TÓPICO | "
                f"{len(grupo)} tweets"
            )
            continue

        topic_id = crear_topic_nuevo(
            cluster_id,
            centroide["embedding"],
            asignaciones
        )

        if topic_id is not None:
            resultados.append({
                "cluster": cluster_id,
                "topic_id": topic_id,
                "total_tweets": len(grupo)
            })

    return resultados


def ejecutar_procesamiento(simular=True):

    print("=" * 70)
    print("NETVORA - PROCESAMIENTO DE TOPICOS")
    print("=" * 70)

    print(
        "MODO:",
        "SIMULACION" if simular else "PRODUCCION"
    )

    # ==========================================
    # 1. OBTENER DATOS
    # ==========================================

    tweets = get_tweet_embedding()

    if not tweets:
        print("No hay tweets nuevos.")
        return

    topics = get_topic_embeddings()
    especiales = get_especial_topics()

    print(f"Tweets disponibles: {len(tweets)}")
    print(f"Topicos existentes: {len(topics)}")

    # ==========================================
    # 2. CLASIFICAR CON REFERENCIAS
    # ==========================================

    asignados, pendientes = clasificar_lote(
        tweets,
        topics,
        especiales
    )

    print("\nCLASIFICACION")
    print(f"Asignados: {len(asignados)}")
    print(f"Pendientes: {len(pendientes)}")

    if not simular and not asignados.empty:

        insert_tweets_topic(asignados)

        ids_especiales = set(especiales.values())

        for topic_id in asignados["topic_id"].unique():

            if int(topic_id) not in ids_especiales:

                recalcular_centroide_topic(
                    int(topic_id)
                )

    if pendientes.empty:
        print("No hay tweets pendientes.")
        return

    # ==========================================
    # 3. DETECTAR CLUSTERS
    # ==========================================

    clusters = detectar_nuevos_clusters(
        pendientes
    )

    if clusters is None or clusters.empty:
        print("No se detectaron clusters.")
        return

    # ==========================================
    # 4. FILTRAR POR CALIDAD
    # ==========================================

    validos = filtrar_clusters_por_calidad(
        clusters
    )

    if validos.empty:
        print("No existen clusters validos.")
        return

    print("\nCLUSTERS VALIDOS")
    print(f"Tweets: {len(validos)}")
    print(f"Clusters: {validos['cluster'].nunique()}")

    # ==========================================
    # 5. CALCULAR CENTROIDES ROBUSTOS
    # ==========================================

    centroides = calcular_centroides_nuevos(
        validos
    )

    if not centroides:
        print("No se calcularon centroides.")
        return

    # ==========================================
    # 6. COMPARAR CON TOPICOS EXISTENTES
    # ==========================================

    topics_actuales = get_topic_embeddings()

    comparacion = comparar_centroides_con_topics(
        centroides,
        topics_actuales
    )

    print("\nCOMPARACION DE TOPICOS")
    print(comparacion.to_string(index=False))

    # ==========================================
    # 7. EVALUAR FUSIONES
    # ==========================================

    fusiones = preparar_clusters_similares(
        validos,
        comparacion,
        topics_actuales
    )

    print("\nFUSIONES")
    print(f"Tweets candidatos: {len(fusiones)}")

    if not simular and not fusiones.empty:

        insert_tweets_topic(fusiones)

        for topic_id in fusiones["topic_id"].unique():

            recalcular_centroide_topic(
                int(topic_id)
            )

    # ==========================================
    # 8. EVALUAR CREACION DE TOPICOS
    # ==========================================

    # Evaluar TODOS los clusters validos.
    # La funcion decide cuales fusionar,
    # crear o dejar pendientes.

    nuevos = crear_todos_los_topics_nuevos(
        comparacion.copy(),
        centroides,
        validos,
        df_fusiones=fusiones,
        simular=simular
    )

    # ==========================================
    # 9. RESULTADOS
    # ==========================================

    print("\n" + "=" * 70)
    print("PROCESAMIENTO FINALIZADO")
    print("=" * 70)

    print(f"Tweets analizados: {len(tweets)}")
    print(f"Asignaciones directas: {len(asignados)}")
    print(f"Tweets candidatos a fusion: {len(fusiones)}")
    print(f"Clusters validos: {validos['cluster'].nunique()}")

    if simular:
        print("SIMULACION: no se modifico MySQL")
    else:
        print(f"Topicos creados: {len(nuevos)}")
        print("PROCESAMIENTO REAL FINALIZADO")

def mostrar_muestras(df, ids_clusters, cantidad=10):

    for cluster_id in ids_clusters:

        grupo = df[
            df["cluster"] == cluster_id
        ].drop_duplicates(subset="tweetid")

        if grupo.empty:
            continue

        print("\n" + "=" * 70)
        print(f"CLUSTER {cluster_id}")
        print(f"TOTAL: {len(grupo)}")
        print("=" * 70)

        muestra = grupo.sample(
            n=min(cantidad, len(grupo)),
            random_state=42
        )

        for i, fila in enumerate(
            muestra.itertuples(index=False), 1
        ):
            print(
                f"{i}. {str(fila.text)[:350]}"
            )
            print("-" * 50)


def calcular_centroide_robusto(grupo):

    grupo = grupo.drop_duplicates(
        subset="tweetid"
    ).copy()

    # Evitar que textos repetidos dominen el centroide.
    textos = (
        grupo["text"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(r"https?://\S+", "", regex=True)
        .str.replace(r"^rt\s+@\w+:\s*", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    grupo["_texto_normalizado"] = textos

    grupo = grupo[
        grupo["_texto_normalizado"] != ""
    ].drop_duplicates(
        subset="_texto_normalizado"
    )

    matriz = np.asarray(
        grupo["embedding"].tolist(),
        dtype=np.float64
    )

    if matriz.size == 0:
        return None, 0, 0.0

    # Descartar vectores inválidos.
    normas = np.linalg.norm(matriz, axis=1)

    validos = (
        np.isfinite(matriz).all(axis=1)
        & np.isfinite(normas)
        & (normas > 0)
    )

    matriz = matriz[validos]
    normas = normas[validos]

    if len(matriz) == 0:
        return None, 0, 0.0

    matriz = matriz / normas[:, None]

    # Centroide inicial utilizando toda la muestra.
    centroide = matriz.mean(axis=0)

    norma = np.linalg.norm(centroide)

    if norma < 1e-12:
        return None, 0, 0.0

    centroide /= norma

    # Refinamiento iterativo.
    for _ in range(3):

        similitudes = matriz @ centroide

        p20, p80 = np.percentile(
            similitudes, [20, 80]
        )

        if p80 - p20 < 1e-8:
            pesos = np.ones(len(matriz))
        else:
            pesos = np.clip(
                (similitudes - p20) / (p80 - p20),
                0.25,
                1.0
            )

        # Los tweets periféricos siguen contribuyendo,
        # pero con menor influencia.
        nuevo_centroide = np.average(
            matriz,
            axis=0,
            weights=pesos
        )

        norma = np.linalg.norm(nuevo_centroide)

        if norma < 1e-12:
            break

        centroide = nuevo_centroide / norma

    similitudes_finales = matriz @ centroide

    return (
        centroide.astype(np.float32).tolist(),
        len(matriz),
        float(np.median(similitudes_finales))
    )



def get_referencias_topics(topic_ids, topics_existentes):

    ids = sorted(set(int(x) for x in topic_ids))

    if not ids:
        return {}

    centroides = {
        int(t["topic_id"]): normalizar_vectores(
            [t["embedding"]]
        )[0]
        for t in topics_existentes
        if int(t["topic_id"]) in ids
    }

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        placeholders = ",".join(["%s"] * len(ids))

        query = f"""
            SELECT
                topic_id,
                tweetid,
                text,
                embedding_vector
            FROM (
                SELECT
                    tt.topic_id,
                    tt.tweetid,
                    t.text,
                    e.embedding_vector,
                    ROW_NUMBER() OVER (
                        PARTITION BY tt.topic_id
                        ORDER BY
                            tt.similarity DESC,
                            tt.tweetid DESC
                    ) AS rn
                FROM topic_tweets tt
                JOIN Tweets t
                    ON t.tweetid = tt.tweetid
                JOIN tweet_embeddings e
                    ON e.tweetid = tt.tweetid
                WHERE tt.topic_id IN ({placeholders})
                  AND t.text IS NOT NULL
            ) AS ranking
            WHERE rn <= %s
        """

        cursor.execute(
            query,
            tuple(ids) + (CANDIDATOS_REFERENCIAS,)
        )

        resultados = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    candidatos = {topic_id: [] for topic_id in ids}

    for fila in resultados:

        topic_id = int(fila["topic_id"])
        centroide = centroides.get(topic_id)

        if centroide is None:
            continue

        try:
            vector = np.asarray(
                json.loads(fila["embedding_vector"]),
                dtype=np.float32
            )

            if vector.shape != centroide.shape:
                continue

            if not np.isfinite(vector).all():
                continue

            norma = np.linalg.norm(vector)

            if norma <= 0:
                continue

            vector = vector / norma

            texto = re.sub(
                r"https?://\S+",
                "",
                fila["text"]
            )

            texto = re.sub(
                r"^RT\s+@\w+:\s*",
                "",
                texto,
                flags=re.IGNORECASE
            )

            texto = " ".join(texto.casefold().split())

            if len(texto) < 20:
                continue

            similitud_centroide = float(
                vector @ centroide
            )

            candidatos[topic_id].append(
                (similitud_centroide, texto, vector)
            )

        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    referencias = {}

    for topic_id, elementos in candidatos.items():

        elementos.sort(
            key=lambda x: x[0],
            reverse=True
        )

        seleccionados = []
        textos_vistos = set()

        for score, texto, vector in elementos:

            if texto in textos_vistos:
                continue

            if seleccionados:

                similitudes = (
                    np.asarray(seleccionados) @ vector
                )

                if np.max(similitudes) >= SIMILITUD_MAX_ENTRE_REFERENCIAS:
                    continue

            seleccionados.append(vector)
            textos_vistos.add(texto)

            if len(seleccionados) >= REFERENCIAS_POR_TOPIC:
                break

        if seleccionados:
            referencias[topic_id] = np.asarray(
                seleccionados,
                dtype=np.float32
            )

        else:
            referencias[topic_id] = np.empty(
                (0, len(centroides[topic_id])),
                dtype=np.float32
            )

    return referencias


def validar_tweet_con_referencias(
    embedding,
    referencias
):

    if referencias is None:
        return False

    if len(referencias) < MIN_REFERENCIAS_TOPIC:
        return False

    vector = normalizar_vectores(
        [embedding]
    )[0]

    if (
        not np.isfinite(vector).all()
        or np.linalg.norm(vector) == 0
        or referencias.shape[1] != len(vector)
    ):
        return False

    similitudes = referencias @ vector

    mejores_5 = np.sort(similitudes)[-5:]

    coincidencias = int(
        np.sum(
            similitudes >= UMBRAL_REFERENCIA
        )
    )

    promedio_top5 = float(
        np.mean(mejores_5)
    )

    return (
        coincidencias >= MIN_COINCIDENCIAS_REFERENCIA
        and promedio_top5 >= UMBRAL_PROMEDIO_TOP5
    )



def obtener_candidatos_topics(embeddings, topics):

    if not topics or not embeddings:
        return [[] for _ in embeddings]

    ids = [int(t["topic_id"]) for t in topics]

    matriz_topics = normalizar_vectores(
        [t["embedding"] for t in topics]
    )

    resultados = []

    for inicio in range(0, len(embeddings), 500):

        lote = normalizar_vectores(
            embeddings[inicio:inicio + 500]
        )

        similitudes = lote @ matriz_topics.T

        for fila in similitudes:

            indices = np.argsort(fila)[
                -CANDIDATOS_POR_TWEET:
            ][::-1]

            candidatos = [
                (ids[i], float(fila[i]))
                for i in indices
                if fila[i] >= UMBRAL_CANDIDATO
            ]

            resultados.append(candidatos)

    return resultados


def seleccionar_topic_preciso(
    embedding,
    candidatos,
    referencias
):

    aprobados = []

    vector = normalizar_vectores([embedding])[0]

    for topic_id, score_centroide in candidatos:

        refs = referencias.get(topic_id)

        if not validar_tweet_con_referencias(
            embedding,
            refs
        ):
            continue

        similitudes = refs @ vector

        top5 = np.sort(similitudes)[-5:]

        score_referencias = float(
            np.mean(top5)
        )

        aprobados.append({
            "topic_id": topic_id,
            "similarity": score_centroide,
            "score_referencias": score_referencias
        })

    if not aprobados:
        return None

    aprobados.sort(
        key=lambda x: x["score_referencias"],
        reverse=True
    )

    # Si dos tópicos obtienen resultados muy
    # similares, no asignar automáticamente.
    if len(aprobados) > 1:

        diferencia = (
            aprobados[0]["score_referencias"]
            - aprobados[1]["score_referencias"]
        )

        if diferencia < MARGEN_REFERENCIAS:
            return None

    return aprobados[0]



def evaluar_novedad_cluster(grupo, topics_existentes):

    if not topics_existentes:
        return True, 0.0

    candidatos = obtener_candidatos_topics(
        grupo["embedding"].tolist(),
        topics_existentes
    )

    ids_topics = {
        topic_id
        for opciones in candidatos
        for topic_id, _ in opciones
    }

    referencias = get_referencias_topics(
        ids_topics,
        topics_existentes
    )

    # Si faltan referencias de un tópico candidato,
    # no podemos descartar una posible coincidencia.
    for topic_id in ids_topics:

        refs = referencias.get(topic_id)

        if refs is None or len(refs) < MIN_REFERENCIAS_TOPIC:
            return False, None

    coincidencias = 0

    for tweet, opciones in zip(
        grupo.itertuples(index=False),
        candidatos
    ):

        resultado = seleccionar_topic_preciso(
            tweet.embedding,
            opciones,
            referencias
        )

        if resultado is not None:
            coincidencias += 1

    fraccion = coincidencias / len(grupo)

    puede_crearse = (
        fraccion < MAX_FRACCION_COINCIDENCIAS_EXISTENTES
    )

    return puede_crearse, fraccion

if __name__ == "__main__":

    # True: no guardar en MySQL
    # False: guardar asignaciones y crear topicos

    MODO_PRUEBA = False

    ejecutar_procesamiento(
        simular=MODO_PRUEBA
    )
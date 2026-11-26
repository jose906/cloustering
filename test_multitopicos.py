import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import main


SUBVENCION = "El Gobierno anunció la eliminación de la subvención al diésel"
BLOQUEOS = "los transportistas anunciaron bloqueos de carreteras"
DOLAR = "el dólar paralelo registró un nuevo incremento"
TWEET_ID = 1850000000000000001


def vector(similitud, eje):
    valores = np.zeros(4, dtype=np.float32)
    valores[0] = similitud
    valores[eje] = np.sqrt(1 - similitud ** 2)
    return valores


def topico(id_, nombre, similitud, eje, activo=1):
    return {
        "topic_id": id_,
        "topic_name": nombre,
        "active": activo,
        "embedding": vector(similitud, eje),
    }


class ClasificacionMultipleTests(unittest.TestCase):
    def setUp(self):
        self.topics = [
            topico(176, "Eliminación de la subvención al diésel", 0.95, 1),
            topico(180, "Bloqueos de transportistas", 0.91, 2),
            topico(99, "Incremento del dólar paralelo", 0.88, 3),
        ]
        self.referencias = {
            topic_id: np.tile(vector(score, 1), (10, 1))
            for topic_id, score in [(176, 0.97), (180, 0.91), (99, 0.90)]
        }

    def clasificar(self, texto, maximo=None):
        tweets = [{"tweetid": TWEET_ID, "text": texto, "embedding": [1, 0, 0, 0]}]
        with ExitStack() as contexto:
            if maximo is not None:
                contexto.enter_context(patch.object(main, "MAX_TOPICS_PER_TWEET", maximo))
            contexto.enter_context(patch.object(main, "SECONDARY_TOPIC_THRESHOLD", 0.84))
            contexto.enter_context(patch.object(main, "get_referencias_topics", return_value=self.referencias))
            contexto.enter_context(redirect_stdout(io.StringIO()))
            return main.clasificar_lote(tweets, self.topics, {"__RESUMEN__": 500})

    def comprobar(self, texto, ids, **config):
        asignados, pendientes = self.clasificar(texto, **config)
        self.assertEqual(asignados["topic_id"].tolist(), ids)
        self.assertEqual(pendientes.empty, bool(ids))
        if ids:
            self.assertEqual(asignados["tweetid"].tolist(), [TWEET_ID] * len(ids))

    def test_una_afirmacion(self):
        self.comprobar(SUBVENCION, [176])

    def test_dos_afirmaciones(self):
        self.comprobar(f"{SUBVENCION}. {BLOQUEOS}.", [176, 180])

    def test_tres_afirmaciones_y_limite_total(self):
        texto = f"{SUBVENCION}. Por otra parte, {BLOQUEOS} y {DOLAR}."
        self.comprobar(texto, [176, 180, 99])
        self.comprobar(texto, [176, 180], maximo=2)
        self.comprobar(texto, [176], maximo=1)

    def test_similitud_por_si_sola_no_agrega_topicos(self):
        self.comprobar(f"{SUBVENCION}. El Congreso reanudó la sesión.", [176])

    def test_margen_estrecho_con_eventos_distintos(self):
        self.referencias[180] = np.tile(vector(0.96, 2), (10, 1))
        self.comprobar(f"{SUBVENCION}. {BLOQUEOS}.", [176, 180])
        self.comprobar(SUBVENCION, [])

    def test_principal_con_menor_coseno_se_sigue_eligiendo_por_referencias(self):
        self.referencias[176], self.referencias[180] = self.referencias[180], self.referencias[176]
        self.comprobar(f"{SUBVENCION}. {BLOQUEOS}.", [180, 176])

    def test_mencion_incidental(self):
        self.comprobar(f"{SUBVENCION}. El ministro mencionó los bloqueos de transportistas.", [176])
        self.comprobar(f"{SUBVENCION}. La ministra habló sobre el incremento del dólar paralelo.", [176])

    def test_publicacion_especial(self):
        self.comprobar(f"Resumen de noticias. {SUBVENCION}. {BLOQUEOS}.", [500])

    def test_pendiente_sin_referencias_compatibles(self):
        self.referencias.clear()
        self.comprobar(f"{SUBVENCION}. {BLOQUEOS}.", [])

    def test_secundario_requiere_referencias_y_umbral(self):
        self.referencias.pop(180)
        texto = f"{SUBVENCION}. {BLOQUEOS}. {DOLAR}."
        self.topics[2]["embedding"] = vector(0.839, 3)
        self.comprobar(texto, [176])

        self.topics[2]["embedding"] = vector(0.845, 3)
        self.referencias[99] = np.tile(vector(0.825, 3), (10, 1))
        self.comprobar(texto, [176, 99])

    def test_conserva_el_margen_con_limite_de_un_topico(self):
        self.referencias[180] = self.referencias[176]
        self.comprobar(f"{SUBVENCION}. {BLOQUEOS}.", [], maximo=1)

    def test_secundarios_inactivos_especiales_y_provisionales(self):
        self.topics[1]["active"] = 0
        for nombre in ("NUEVO_CLUSTER_9_20261125", "__RESUMEN__"):
            self.topics[2]["topic_name"] = nombre
            self.comprobar(f"{SUBVENCION}. {BLOQUEOS}. {DOLAR}.", [176])

    def test_conserva_solo_un_topico_para_un_evento_duplicado(self):
        self.topics[1] = topico(180, "Retiro del subsidio al combustible", 0.91, 1)
        self.comprobar(f"{SUBVENCION}. El retiro del subsidio al combustible comenzó.", [176])


class PersistenciaMultipleTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()
        self.cursor = self.connection.cursor.return_value
        self.cursor.fetchall.side_effect = [
            [{"tweetid": TWEET_ID, "asignacion": 5}, {"tweetid": TWEET_ID + 1, "asignacion": None}],
            [],
        ]
        self.asignaciones = pd.DataFrame(
            [(176, TWEET_ID, 0.92), (180, TWEET_ID + 1, 0.90),
             (99, TWEET_ID + 1, 0.94), (99, TWEET_ID + 1, 0.88)],
            columns=["topic_id", "tweetid", "similarity"],
        )

    def test_incremental_en_orden_reintento_y_conteo_por_topico(self):
        with patch.object(main, "get_db_connection", return_value=self.connection):
            self.assertEqual(main.insert_tweets_topic(self.asignaciones), {180})
            insercion, conteo = self.cursor.executemany.call_args_list
            self.assertEqual(insercion.args[1], [(180, TWEET_ID + 1, 0.90), (99, TWEET_ID + 1, 0.94)])
            self.assertIn("COUNT(DISTINCT tt.tweetid)", conteo.args[0])
            self.assertEqual(conteo.args[1], [(99,), (180,)])
            self.assertIn("FOR UPDATE", self.cursor.execute.call_args_list[0].args[0])

            self.cursor.fetchall.side_effect = [[
                {"tweetid": TWEET_ID, "asignacion": 5},
                {"tweetid": TWEET_ID + 1, "asignacion": 6},
                {"tweetid": TWEET_ID + 1, "asignacion": 7},
            ]]
            self.cursor.executemany.reset_mock()
            self.assertEqual(main.insert_tweets_topic(self.asignaciones), set())
            self.cursor.executemany.assert_not_called()
        self.assertEqual(self.connection.commit.call_count, 2)

    def test_una_falla_de_integridad_revierte_el_lote(self):
        fallo = main.mysql.connector.IntegrityError("Clave foránea inválida", errno=1452)
        self.cursor.executemany.side_effect = fallo
        with patch.object(main, "get_db_connection", return_value=self.connection):
            with self.assertRaises(main.mysql.connector.IntegrityError):
                main.insert_tweets_topic(self.asignaciones)
        self.connection.rollback.assert_called_once()
        self.connection.commit.assert_not_called()

    def test_centroides_y_referencias_usan_solo_la_primera_relacion(self):
        self.cursor.fetchall.side_effect = None
        self.cursor.fetchall.return_value = []
        self.cursor.fetchone.return_value = {"embedding_vector": "[1, 0, 0, 0]"}
        with patch.object(main, "get_db_connection", return_value=self.connection):
            main.recalcular_centroide_topic(99, simular=True)
            main.get_referencias_topics([99], [topico(99, "Incremento del dólar", 0.88, 3)])
        for consulta in self.cursor.execute.call_args_list[1:]:
            self.assertIn("anterior.id < tt.id", consulta.args[0])

    def test_referencias_descartan_textos_y_vectores_repetidos(self):
        self.cursor.fetchall.side_effect = None
        self.cursor.fetchall.return_value = [
            {"topic_id": 99, "text": "El dólar registra un aumento en Bolivia", "embedding_vector": "[1, 0, 0, 0]"},
            {"topic_id": 99, "text": "RT @medio: El dólar registra un aumento en Bolivia", "embedding_vector": "[0, 1, 0, 0]"},
            {"topic_id": 99, "text": "Publicación con un embedding repetido", "embedding_vector": "[1, 0, 0, 0]"},
            {"topic_id": 99, "text": "Otra publicación sobre el aumento del dólar", "embedding_vector": "[0, 0, 0, 1]"},
            {"topic_id": 99, "text": "Publicación con un embedding incompatible", "embedding_vector": "[1, 0]"},
        ]
        with patch.object(main, "get_db_connection", return_value=self.connection):
            refs = main.get_referencias_topics([99], [topico(99, "Aumento del dólar", 1, 3)])
        np.testing.assert_array_equal(refs[99], [[1, 0, 0, 0], [0, 0, 0, 1]])

    def test_el_proceso_recalcula_solo_principales_insertados(self):
        with ExitStack() as contexto:
            contexto.enter_context(patch.object(main, "get_tweet_embedding", return_value=[{"tweetid": TWEET_ID}]))
            contexto.enter_context(patch.object(main, "get_especial_topics", return_value={}))
            contexto.enter_context(patch.object(main, "get_topic_embeddings", return_value=[]))
            contexto.enter_context(patch.object(main, "clasificar_lote", return_value=(self.asignaciones, pd.DataFrame())))
            contexto.enter_context(patch.object(main, "get_db_connection", return_value=self.connection))
            recalcular = contexto.enter_context(patch.object(main, "recalcular_centroide_topic"))
            contexto.enter_context(redirect_stdout(io.StringIO()))
            main.classify_tweets()
        recalcular.assert_called_once_with(180)


if __name__ == "__main__":
    unittest.main()

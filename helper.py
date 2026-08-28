import re

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

def detectar_topic_especial(texto):
    texto_original = str(texto).strip()
    texto_lower = texto_original.lower()

    # 1. Resumen de noticias
    if (
        "resumen de noticias" in texto_lower
        or "resumen informativo" in texto_lower
        or "resumen de la jornada" in texto_lower
    ):
        return "__RESUMEN__"

    # 2. Portada de periódico
    if (
        "portada de la edición impresa" in texto_lower
        or "portada de hoy" in texto_lower
        or "nuestra portada" in texto_lower
        or "te presentamos la edición impresa" in texto_lower
    ):
        return "__PORTADA__"

    # 3. E-paper o edición digital
    if (
        "epaper" in texto_lower
        or "e-paper" in texto_lower
        or "edición digital" in texto_lower
        or "edicion digital" in texto_lower
    ):
        return "__EPAPER__"

    # 4. Programación
    if (
        "programación" in texto_lower
        or "programacion" in texto_lower
        or "en vivo:" in texto_lower
        or "no te pierdas" in texto_lower
        or "sintoniza" in texto_lower
    ):
        return "__PROGRAMACION__"

    # 5. Promoción
    if (
        "promoción" in texto_lower
        or "promocion" in texto_lower
        or "participa y gana" in texto_lower
        or "sorteo" in texto_lower
        or "suscríbete" in texto_lower
        or "suscribete" in texto_lower
    ):
        return "__PROMOCION__"

    # 6. Saludos
    if (
        texto_lower.startswith("buenos días")
        or texto_lower.startswith("buenos dias")
        or texto_lower.startswith("buenas tardes")
        or texto_lower.startswith("buenas noches")
        or texto_lower.startswith("feliz lunes")
        or texto_lower.startswith("feliz martes")
        or texto_lower.startswith("feliz miércoles")
        or texto_lower.startswith("feliz miercoles")
        or texto_lower.startswith("feliz jueves")
        or texto_lower.startswith("feliz viernes")
    ):
        return "__SALUDO__"

    # 7. Solo link
    texto_sin_urls = URL_REGEX.sub("", texto_original)
    texto_sin_urls = re.sub(r"[\s\W_]+", "", texto_sin_urls)

    if URL_REGEX.search(texto_original) and texto_sin_urls == "":
        return "__SOLO_LINK__"

    return None
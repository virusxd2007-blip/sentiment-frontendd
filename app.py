from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Descargar recursos solo si es necesario
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Inicializar analizador
sia = SentimentIntensityAnalyzer()

# Ajustar léxico para español
spanish_words = {
    'bueno': 2.0, 'malo': -2.0, 'excelente': 3.0, 'terrible': -3.0,
    'genial': 2.5, 'horrible': -2.5, 'amor': 2.0, 'odio': -2.0,
    'feliz': 2.0, 'triste': -2.0, 'bien': 1.5, 'mal': -1.5,
    'positivo': 2.0, 'negativo': -2.0, 'increíble': 3.0, 'pésimo': -3.0
}
sia.lexicon.update(spanish_words)

def limpiar_texto(texto):
    """Limpia el texto: minúsculas, quita URLs, menciones, hashtags y símbolos."""
    if not texto or len(texto.strip()) == 0:
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)  # Elimina URLs
    texto = re.sub(r'@\w+', '', texto)            # Elimina @usuario
    texto = re.sub(r'#', '', texto)               # Elimina símbolo #
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)  # Solo letras y espacios
    return ' '.join(texto.split())                # Normaliza espacios

def extraer_palabras_clave(texto):
    """Extrae palabras clave (sin stopwords del español)."""
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words('spanish'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('spanish'))
    
    palabras = texto.lower().split()
    return [p for p in palabras if p not in stop_words and len(p) > 3]

def analizar_sentimiento(texto):
    """Analiza un solo comentario y devuelve resultado estructurado."""
    if not texto or len(texto.strip()) == 0:
        return {"sentimiento": "Neutro", "confianza": 0.0, "texto_limpio": "", "detalle": {}}
    
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return {"sentimiento": "Neutro", "confianza": 0.0, "texto_limpio": "", "detalle": {}}
    
    scores = sia.polarity_scores(texto_limpio)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentimiento = "Positivo"
    elif compound <= -0.05:
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutro"
    
    confianza = round(abs(compound) * 100, 2)
    return {
        "sentimiento": sentimiento,
        "confianza": confianza,
        "texto_limpio": texto_limpio,
        "detalle": scores
    }

def analizar_comentarios_masivo(lista):
    """Analiza múltiples comentarios y devuelve estadísticas."""
    resultados = []
    contadores = {"Positivo": 0, "Negativo": 0, "Neutro": 0}
    todas_palabras = []
    
    for texto in lista:
        res = analizar_sentimiento(texto)
        resultados.append(res)
        contadores[res["sentimiento"]] += 1
        todas_palabras.extend(extraer_palabras_clave(res["texto_limpio"]))
    
    # Contar palabras más frecuentes
    from collections import Counter
    top_palabras = [{"palabra": p, "frecuencia": f} for p, f in Counter(todas_palabras).most_common(10)]
    
    return {
        "estadisticas": {
            "contadores": contadores,
            "total": len(lista),
            "top_palabras": top_palabras
        }
    }

# Crear app Flask
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """Ruta raíz para evitar 404 en Fly.io/Vercel."""
    return "¡Backend de Análisis de Sentimientos funcionando! 🚀"

@app.route('/analizar', methods=['POST'])
def analizar():
    data = request.get_json()
    texto = data.get("texto", "")
    resultado = analizar_sentimiento(texto)
    return jsonify(resultado)

@app.route('/analizar-multiple', methods=['POST'])
def analizar_multiple():
    data = request.get_json()
    comentarios = data.get("comentarios", [])
    resultado = analizar_comentarios_masivo(comentarios)
    return jsonify(resultado)

# Handler para Vercel (opcional, pero no hace daño)
def handler(request):
    return app

# Para Fly.io/Render: esto se ignora si usas gunicorn
if __name__ == '__main__':
    app.run(debug=True)
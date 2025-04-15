import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import openai
from dotenv import load_dotenv
import os
import re
from difflib import get_close_matches
import json

# Cargar variables de entorno
load_dotenv('env.local.download')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No se encontró la clave de API de OpenAI en el archivo env.local.download")

# Configurar la API key de OpenAI
openai.api_key = openai_api_key
print(f"API Key cargada correctamente: {openai_api_key[:5]}...")

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load('es_core_news_lg')
    print("Modelo de spaCy cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo de spaCy: {str(e)}")
    print("Intentando descargar el modelo...")
    os.system('python -m spacy download es_core_news_lg')
    nlp = spacy.load('es_core_news_lg')

# Definir categorías y tipos
CATEGORIAS = [
    'Datos_Pasajero', 'Website', 'Proceso_Pago', 'Discount_Club',
    'Promociones', 'Precios', 'Disponibilidad_Vuelo', 'Aeropuerto',
    'Seats', 'Equipaje', 'Cambios_Devoluciones', 'Servicio_General'
]

def limpiar_texto(texto):
    """Limpia y normaliza el texto."""
    if not isinstance(texto, str):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    
    # Eliminar espacios extra
    texto = ' '.join(texto.split())
    
    return texto

def limpiar_respuesta_json(respuesta):
    """Limpia la respuesta JSON de marcadores de código Markdown."""
    # Eliminar marcadores de código Markdown
    respuesta = re.sub(r'```json\s*', '', respuesta)
    respuesta = re.sub(r'```\s*$', '', respuesta)
    
    # Eliminar espacios en blanco al inicio y final
    respuesta = respuesta.strip()
    
    return respuesta

def obtener_categorizacion_gpt(comentario):
    """Obtiene la categorización usando la API de ChatGPT."""
    try:
        prompt = f"""
        Analiza el siguiente comentario y categorízalo según estas categorías:
        {', '.join(CATEGORIAS)}
        
        Comentario: {comentario}
        
        INSTRUCCIONES ESPECIALES:
        1. Para la categoría 'Cambios_Devoluciones', incluye todo lo relacionado con:
           - Cancelaciones de vuelos
           - Suspensiones de vuelos
           - Cambios de horario
           - Cambios de día
           - Solicitudes de devoluciones
           - Cualquier modificación o cancelación de reservas
        
        2. Para el campo 'tipo', proporciona una descripción CONCISA que:
           - Sea breve y directa (máximo 4-5 palabras)
           - Capture la esencia del comentario
           - No incluya detalles innecesarios (países, número de vuelos, etc.)
           - Mantenga consistencia en descripciones similares
           
           Ejemplos de tipos concisos:
           - "Altos precios de retorno"
           - "Mejorar interfaz página"
           - "Buena experiencia"
           - "Problema en pago"
           - "Solicitud de devolución"
           - "Cancelación de vuelo"
           - "Problema con equipaje"
           - "Sugerencia de precios"
           - "Consulta de horarios"
           - "Problema en check-in"
           
           IMPORTANTE: 
           - El tipo debe ser breve y directo
           - No incluir información que se pueda obtener de otros campos
           - Mantener consistencia en descripciones similares
           - No forzar connotación negativa
           - Si el comentario es positivo, reflejarlo en el tipo
        
        Responde en formato JSON con los siguientes campos:
        - categoria: (una de las categorías listadas)
        - subcategoria: (si aplica, o null si no aplica)
        - tipo: (descripción breve y concisa del contenido del comentario)
        - confianza: (número entre 0 y 1)
        
        IMPORTANTE: Responde SOLO con el JSON, sin marcadores de código ni texto adicional.
        """
        
        print(f"Enviando solicitud a GPT para comentario: {comentario[:50]}...")
        
        # Usar el modelo gpt-4o-mini
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de comentarios de clientes. Responde SOLO con JSON válido, sin marcadores de código ni texto adicional."},
                {"role": "user", "content": prompt}
            ]
        )
        
        respuesta = response.choices[0].message.content
        print(f"Respuesta recibida: {respuesta}")
        
        # Limpiar la respuesta de marcadores de código Markdown
        respuesta_limpia = limpiar_respuesta_json(respuesta)
        print(f"Respuesta limpia: {respuesta_limpia}")
        
        # Verificar que la respuesta es un JSON válido
        try:
            # Intentar parsear como JSON
            resultado = json.loads(respuesta_limpia)
            
            # Si es un array, tomar el primer elemento
            if isinstance(resultado, list) and len(resultado) > 0:
                resultado = resultado[0]
                
            return json.dumps(resultado)
        except json.JSONDecodeError as e:
            print(f"Error al parsear JSON: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error al categorizar con GPT: {str(e)}")
        return None

def encontrar_columna_comentario(df):
    """Encuentra la columna de comentarios usando coincidencia aproximada."""
    # Lista de posibles nombres de columnas de comentarios
    nombres_posibles = [
        'Comentario', 'comentario', 'COMENTARIO', 'Comment', 'comment', 'COMMENT',
        'Comentairo', 'Comentarios', 'comentarios', 'COMENTARIOS',
        'Texto', 'texto', 'TEXTO', 'Text', 'text', 'TEXT',
        'Respuesta', 'respuesta', 'RESPUESTA', 'Response', 'response', 'RESPONSE'
    ]
    
    # Primero buscar coincidencia exacta
    for nombre in nombres_posibles:
        if nombre in df.columns:
            return nombre
    
    # Si no hay coincidencia exacta, buscar coincidencia aproximada
    todas_columnas = df.columns.tolist()
    for nombre in nombres_posibles:
        coincidencias = get_close_matches(nombre, todas_columnas, n=1, cutoff=0.8)
        if coincidencias:
            print(f"Se encontró una coincidencia aproximada: '{coincidencias[0]}' para '{nombre}'")
            return coincidencias[0]
    
    return None

def procesar_archivo_entrenamiento(archivo):
    """Procesa el archivo de entrenamiento y extrae las características."""
    print(f"Procesando archivo: {archivo}")
    df = pd.read_excel(archivo)
    print(f"\nColumnas en {archivo}:")
    print(df.columns.tolist())
    
    # Buscar la columna de comentarios
    columna_comentario = encontrar_columna_comentario(df)
    
    if columna_comentario is None:
        raise ValueError(f"No se encontró la columna de comentarios en {archivo}. Columnas disponibles: {df.columns.tolist()}")
    
    print(f"Usando columna '{columna_comentario}' para comentarios")
    df['Comentario_Limpio'] = df[columna_comentario].apply(limpiar_texto)
    return df, columna_comentario

def main():
    try:
        # Cargar datos de entrenamiento
        print("Procesando archivos de entrenamiento...")
        df_ejemplo1, col_comentario1 = procesar_archivo_entrenamiento('Categorization_Example.xlsx')
        df_ejemplo2, col_comentario2 = procesar_archivo_entrenamiento('Categorization_Example_2.xlsx')
        
        # Cargar datos de prueba
        print("\nProcesando archivo de prueba...")
        df_test = pd.read_excel('Test_2.xlsx')
        print("Columnas en Test_2.xlsx:")
        print(df_test.columns.tolist())
        
        # Buscar la columna de comentarios en el archivo de prueba
        columna_comentario = encontrar_columna_comentario(df_test)
        
        if columna_comentario is None:
            raise ValueError(f"No se encontró la columna de comentarios en Test_2.xlsx. Columnas disponibles: {df_test.columns.tolist()}")
        
        print(f"Usando columna '{columna_comentario}' para comentarios en archivo de prueba")
        df_test['Comentario_Limpio'] = df_test[columna_comentario].apply(limpiar_texto)
        
        # Verificar que hay datos para procesar
        if len(df_test) == 0:
            raise ValueError("El archivo de prueba está vacío")
        
        print(f"Procesando {len(df_test)} comentarios...")
        
        # Procesar cada comentario del archivo de prueba
        resultados = []
        for idx, row in df_test.iterrows():
            print(f"Procesando comentario {idx + 1}/{len(df_test)}")
            
            # Verificar que el comentario no está vacío
            if pd.isna(row[columna_comentario]) or row[columna_comentario].strip() == "":
                print(f"Comentario {idx + 1} está vacío, saltando...")
                continue
                
            # Obtener categorización de GPT
            categorizacion = obtener_categorizacion_gpt(row[columna_comentario])
            
            if categorizacion:
                try:
                    # Convertir la respuesta de GPT a diccionario
                    cat_dict = json.loads(categorizacion)
                    
                    # Verificar que todos los campos requeridos están presentes
                    campos_requeridos = ['categoria', 'subcategoria', 'tipo', 'confianza']
                    for campo in campos_requeridos:
                        if campo not in cat_dict:
                            print(f"Error: Falta el campo '{campo}' en la respuesta de GPT")
                            continue
                    
                    resultados.append({
                        'PNR': row['pnr'],
                        'Comentario': row[columna_comentario],
                        'Categoria': cat_dict['categoria'],
                        'Subcategoria': cat_dict['subcategoria'],
                        'Tipo': cat_dict['tipo'],
                        'Confianza': cat_dict['confianza']
                    })
                    print(f"Comentario {idx + 1} procesado correctamente")
                except Exception as e:
                    print(f"Error al procesar la categorización para el comentario {idx + 1}: {str(e)}")
            else:
                print(f"No se pudo obtener categorización para el comentario {idx + 1}")
        
        # Verificar que hay resultados para guardar
        if len(resultados) == 0:
            raise ValueError("No se obtuvieron resultados para guardar")
        
        # Crear DataFrame con resultados
        df_resultados = pd.DataFrame(resultados)
        print(f"Se obtuvieron {len(df_resultados)} resultados")
        
        # Generar nombre de archivo con timestamp
        from datetime import datetime
        nombre_archivo = f'Resultados_Categorizacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        try:
            # Guardar resultados en archivo nuevo
            df_resultados.to_excel(nombre_archivo, index=False)
            print(f"Resultados guardados exitosamente en '{nombre_archivo}'")
        except Exception as e:
            print(f"Error al guardar los resultados como Excel: {str(e)}")
            # Intentar guardar como CSV como último recurso
            try:
                nombre_csv = f'Resultados_Categorizacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                df_resultados.to_csv(nombre_csv, index=False)
                print(f"Los resultados se guardaron como CSV en '{nombre_csv}'")
            except Exception as csv_error:
                print(f"No se pudo guardar los resultados ni como Excel ni como CSV: {str(csv_error)}")
    
    except Exception as e:
        print(f"Error en el proceso: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
# Categorizador de Comentarios CSAT

Este proyecto implementa un sistema de categorización automática de comentarios de CSAT utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## Características

- Categorización automática de comentarios
- Procesamiento de lenguaje natural en español
- Análisis de sentimiento
- Extracción de características usando TF-IDF
- Clasificación usando Random Forest
- Integración con OpenAI GPT para categorización avanzada

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`
- API Key de OpenAI

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/JeffReynaud/Categorizador_API.git
cd Categorizador_API
```

2. Crear y activar el entorno virtual:
```bash
conda create -n milka python=3.8
conda activate milka
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
- Crear un archivo `env.local.download` con tu API key de OpenAI:
```
OPENAI_API_KEY=tu_api_key_aquí
```

## Uso

1. Preparar los archivos de datos:
   - `Categorization_Example.xlsx`: Datos de entrenamiento
   - `Test_1.xlsx`: Datos de prueba

2. Ejecutar el script:
```bash
python categorizacion_comentarios.py
```

3. Los resultados se guardarán en `Resultados_Categorizacion.xlsx`

## Estructura del Proyecto

- `categorizacion_comentarios.py`: Script principal
- `requirements.txt`: Dependencias del proyecto
- `setup.sh`: Script de configuración
- Archivos de datos:
  - `Categorization_Example.xlsx`
  - `Categorization_Example_2.xlsx`
  - `Test_1.xlsx`
  - `Test_2.xlsx`

## Autor

Jefferson Reynaud
- Email: jreynaud@hotmail.cl

## Licencia

Este proyecto está bajo la Licencia MIT. 
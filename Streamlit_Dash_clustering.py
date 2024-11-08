import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import gdown

st.title('Higher Education in Ecuador')

import streamlit as st
import pandas as pd
import gdown

@st.cache_data
def load_data(url):
    output = "base_matricula_datosabiertos.xlsx"
    gdown.download(url, output, quiet=False)
    return pd.read_excel(output, engine='openpyxl')

def preprocess_data(df):
    # Realiza todos los reemplazos y filtrados de datos aquí
    df['MODALIDAD'] = df['MODALIDAD'].replace(['HIBRIDA', 'DUAL'], 'SEMIPRESENCIAL')
    df['NIVEL_FORMACIÓN'] = df['NIVEL_FORMACIÓN'].replace(['TERCER NIVEL O PREGRADO'], 'PREGRADO')
    df['NIVEL_FORMACIÓN'] = df['NIVEL_FORMACIÓN'].replace(['CUARTO NIVEL O POSGRADO'], 'POSGRADO')
    df['CAMPO_AMPLIO'] = df['CAMPO_AMPLIO'].replace(['CIENCIAS SOCIALES, PERIODISMO, INFORMACION Y DERECHO'], 'CIENCIAS SOCIALES Y DERECHO')
    df['CAMPO_AMPLIO'] = df['CAMPO_AMPLIO'].replace(['AGRICULTURA, SILVICULTURA, PESCA Y VETERINARIA'], 'AGRICULTURA Y VETERINARIA')
    df['CAMPO_AMPLIO'] = df['CAMPO_AMPLIO'].replace(['CIENCIAS NATURALES, MATEMATICAS Y ESTADISTICA'], 'CIENCIAS NATURALES Y MATEMATICAS')
    df['CAMPO_AMPLIO'] = df['CAMPO_AMPLIO'].replace(['INGENIERIA, INDUSTRIA Y CONSTRUCCION'], 'INGENIERIA E INDUSTRIA')
    df['CAMPO_AMPLIO'] = df['CAMPO_AMPLIO'].replace(['TECNOLOGIAS DE LA INFORMACION Y LA COMUNICACION (TIC)'], 'TECNOLOGIAS DE LA INFORMACION')
    df['TIPO_FINANCIAMIENTO'] = df['TIPO_FINANCIAMIENTO'].replace(['PARTICULAR COFINANCIADA'], 'PARTICULAR')
    df['TIPO_FINANCIAMIENTO'] = df['TIPO_FINANCIAMIENTO'].replace(['PARTICULAR AUTOFINANCIADA'], 'PARTICULAR')
    df = df[(df["CAMPO_AMPLIO"] != "NO_REGISTRA") & 
            (df["PROVINCIA_RESIDENCIA"] != "NO_REGISTRA") &
            (df["PROVINCIA_RESIDENCIA"] != "ZONAS NO DELIMITADAS") &
            (df["NIVEL_FORMACIÓN"] != "TERCER NIVEL TECNICO-TECNOLOGICO SUPERIOR")
            (df["AÑO"] == 2022)]
    return df

# URL de descarga de Google Drive
url = "https://drive.google.com/uc?id=1M3QXxjoUjI5-6bYRZMb95BJkI0i4jT5Z"

# Cargar y procesar datos
df_matricula = load_data(url)
df_matriculas = preprocess_data(df_matricula)

# In[121]:


import pandas as pd
import plotly.express as px
import plotly.colors as pc

# Definimos una paleta de colores personalizada
color_sequence = [ 'rgba(255, 215, 0, 0.3)', 'rgba(128, 128, 128, 0.5)', 'white']

# Agrupamos los datos para crear el gráfico sunburst
sunburst_data = df_matriculas.groupby(['TIPO_FINANCIAMIENTO', 'NIVEL_FORMACIÓN', 'MODALIDAD', 'CAMPO_AMPLIO'])['CODIGO_CARRERA'].nunique().reset_index()
sunburst_data.rename(columns={'CODIGO_CARRERA': 'Cantidad de Carreras'}, inplace=True)

# Identificamos el top 3 de 'CAMPO_AMPLIO' por cantidad de carreras
top_3_campo_amplio = sunburst_data.groupby('CAMPO_AMPLIO')['Cantidad de Carreras'].sum().nlargest(3).index

# Agrupamos las categorías fuera del top 3 en 'Otros'
sunburst_data['CAMPO_AMPLIO'] = sunburst_data['CAMPO_AMPLIO'].apply(lambda x: x if x in top_3_campo_amplio else 'Otros')

# Volvemos a agrupar para consolidar la categoría 'Otros'
sunburst_data = sunburst_data.groupby(['TIPO_FINANCIAMIENTO', 'NIVEL_FORMACIÓN', 'MODALIDAD', 'CAMPO_AMPLIO'])['Cantidad de Carreras'].sum().reset_index()

# Añadimos una columna ficticia para representar el total en el centro
sunburst_data['Total'] = 'CARRERAS/PROGRAMAS'

# Creamos el gráfico sunburst con el mapa de colores personalizado
fig = px.sunburst(
    sunburst_data,
    path=['Total', 'TIPO_FINANCIAMIENTO', 'NIVEL_FORMACIÓN', 'MODALIDAD', 'CAMPO_AMPLIO'],  # Se agrega 'Total' como raíz del gráfico
    values='Cantidad de Carreras',
    color='TIPO_FINANCIAMIENTO',  # Colorea por la categoría TIPO_FINANCIAMIENTO
    color_discrete_map=None,  # No se usa el mapa de colores aquí
    color_discrete_sequence=color_sequence,  # Se usa la secuencia de colores personalizada
    title="Programs/Courses",
    
)

#Mostrar gráfico
st.plotly_chart(fig)

# Actualizamos los detalles del gráfico
fig.update_traces(
    textinfo='label+percent entry',
    hoverinfo='label+percent entry',
    insidetextorientation='radial',
    textfont_size=12
)

# Agregamos una anotación en el centro para mostrar solo el número total de carreras
total_carreras = sunburst_data['Cantidad de Carreras'].sum()
fig.update_layout(
    title_x=0.5,  # Centra el título del gráfico
    annotations=[{
        'text': f'{total_carreras}',  # Muestra solo el número en el centro
        'x': 0.5,
        'y': 0.48,
        'font_size': 24,
        'showarrow': False,
        'font_color': 'black'
    }],
    paper_bgcolor='white',
    font=dict(size=20),
    margin=dict(t=50, l=50, r=50, b=50),
    width=1000,
    height=1000
)

fig.write_image("sunburst_oferta_carreras.png")

# Mostramos el gráfico
fig.show()



# In[122]:


import pandas as pd
import plotly.express as px

# Definimos una paleta de colores personalizada
color_sequence = [ 'rgba(255, 215, 0, 0.3)', 'rgba(128, 128, 128, 0.5)', 'white']

# Agrupamos los datos para crear el gráfico sunburst usando la columna "tot" para el total de estudiantes
sunburst_data = df_matriculas.groupby(['TIPO_FINANCIAMIENTO', 'NIVEL_FORMACIÓN', 'MODALIDAD', 'CAMPO_AMPLIO'])['tot'].sum().reset_index()
sunburst_data.rename(columns={'tot': 'Total Estudiantes'}, inplace=True)

# Añadimos una columna ficticia para representar el total en el centro
sunburst_data['Total'] = 'ESTUDIANTES'

# Creamos el gráfico sunburst con el mapa de colores personalizado
fig = px.sunburst(
    sunburst_data,
    path=['Total', 'TIPO_FINANCIAMIENTO', 'NIVEL_FORMACIÓN', 'MODALIDAD', 'CAMPO_AMPLIO'],  # Se agrega 'Total' como raíz del gráfico
    values='Total Estudiantes',
    color='TIPO_FINANCIAMIENTO',  # Colorea por la categoría TIPO_FINANCIAMIENTO
    color_discrete_map=None,  # No se usa el mapa de colores aquí
    color_discrete_sequence=color_sequence,  # Se usa la secuencia de colores personalizada
    title="Students distribution"
)

st.plotly_chart(fig)

# Actualizamos los detalles del gráfico
fig.update_traces(
    textinfo='label+percent entry',
    hoverinfo='label+percent entry',
    insidetextorientation='radial',
    textfont_size=12
)

# Agregamos una anotación en el centro para mostrar solo el número total de estudiantes
total_estudiantes = sunburst_data['Total Estudiantes'].sum()
fig.update_layout(
    title_x= 0.5,
    annotations=[{
        'text': f'{total_estudiantes}',  # Muestra solo el número en el centro
        'x': 0.5,
        'y': 0.48,
        'font_size': 24,
        'showarrow': False,
        'font_color': 'black',
        'xanchor': 'center',
        'yanchor': 'middle'
    }],
    paper_bgcolor='white',
    font=dict(size=20),
    margin=dict(t=50, l=50, r=50, b=50),
    width=1000,
    height=1000
)
fig.write_image("sunburst_estudiantes.png")
# Mostramos el gráfico
fig.show()


# In[123]:


import pandas as pd
import matplotlib.pyplot as plt
import squarify  # Biblioteca para treemaps

# Agrupamos por provincia y obtenemos el top 10
provincia_totales = df_matriculas.groupby('PROVINCIA_RESIDENCIA')['tot'].sum().reset_index()
top_10_provincias = provincia_totales.nlargest(10, 'tot')

# Calculamos la suma de las demás provincias y creamos la fila "Otros"
otros = pd.DataFrame({
    'PROVINCIA_RESIDENCIA': ['Otros'],
    'tot': [provincia_totales['tot'].sum() - top_10_provincias['tot'].sum()]
})

# Concatenamos el top 10 con la categoría "Otros" y ordenamos de mayor a menor
treemap_data = pd.concat([top_10_provincias, otros], ignore_index=True).sort_values(by='tot', ascending=False)

# Definimos la paleta de colores en formato hexadecimal para cada rango
colors = []
for value in treemap_data['tot']:
    if value > 100000:
        colors.append('#D4AF37')  # Dorado pastel para > 100000
    elif 30000 <= value <= 100000:
        colors.append('#808080')  # Gris oscuro pastel para 100000 >= valor >= 30000
    else:
        colors.append('#D3D3D3')  # Gris claro pastel para < 30000

# Crear el treemap
# plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=treemap_data['tot'],
    label=treemap_data['PROVINCIA_RESIDENCIA'] + "\n" + treemap_data['tot'].apply(lambda x: f'{x:,}'),
    color=colors,
    alpha=0.8,
    text_kwargs={'fontsize': 12, 'color': 'black'}
)

# Títulos y ajustes
plt.title("Estudiantes por Provincia de residencia (Top 10)", fontsize=16)
plt.axis('off')  # Ocultar los ejes

# Guardar el gráfico
plt.savefig("top_10_estudiantes_provincia_treemap.png", format="png", dpi=300)

# Mostrar el gráfico
fig.show()

# In[124]:

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Suponiendo que ya tienes el dataframe `df_matriculas`
@st.cache_data
def get_tot_values_by_province(df):
    df_grouped = df.groupby(['PROVINCIA_RESIDENCIA', 'CAMPO_AMPLIO'])['tot'].sum().reset_index()
    provincias = df_grouped["PROVINCIA_RESIDENCIA"].unique().tolist()
    secondary_nodes = df_grouped["CAMPO_AMPLIO"].unique().tolist()
    tot_values_by_province = {
        provincia: df_grouped[df_grouped['PROVINCIA_RESIDENCIA'] == provincia]
            .set_index('CAMPO_AMPLIO')['tot']
            .reindex(secondary_nodes, fill_value=0)
            .tolist()
        for provincia in provincias
    }
    return tot_values_by_province, provincias, secondary_nodes

tot_values_by_province, provincias, secondary_nodes = get_tot_values_by_province(df_matriculas)

# Crear el selector de provincia con Streamlit
provincia = st.selectbox('Province:', provincias, index=13)  # Valor inicial

# Función para dividir el texto en dos líneas si es muy largo y añadir el valor de 'tot'
def split_text_with_tot(text, tot_value, max_length=15):
    words = text.split()
    half = len(words) // 2
    if len(text) > max_length and len(words) > 1:
        label = "\n".join([" ".join(words[:half]), " ".join(words[half:])])
    else:
        label = text
    # Añadir el valor de 'tot' en una nueva línea
    return f"{label}\nEstudiantes: {tot_value}"

# Función para actualizar el grafo según la provincia seleccionada
def update_graph(provincia):
    # Obtener valores de 'tot' para la provincia seleccionada
    tot_values = tot_values_by_province[provincia]
    
    # Crear el grafo
    G = nx.Graph()
    G.add_node(provincia)
    node_sizes = [5000] + [value * 1 for value in tot_values]  # Tamaño mayor para el nodo raíz, reducir secundarios
    
    for i, node in enumerate(secondary_nodes):
        G.add_node(node)
        G.add_edge(provincia, node)

    # Colores para el nodo raíz y nodos secundarios
    node_colors = ["#1A237E"] + ["#D3D3D3"] * len(secondary_nodes)  # Azul oscuro para el nodo raíz y gris claro para secundarios

    # Crear etiquetas de nodos con texto dividido y con valor de 'tot'
    labels = {node: split_text_with_tot(node, tot_values[i]) for i, node in enumerate(secondary_nodes)}
    labels[provincia] = provincia  # Etiqueta sin cambio para el nodo raíz

    # Dibujar el grafo sin etiquetas en nx.draw
    plt.figure(figsize=(14, 12))
    pos = nx.shell_layout(G, [list(G.neighbors(provincia)), [provincia]])  # Usar shell layout para disposición de estrella
    nx.draw(G, pos, node_size=node_sizes, font_size=8, node_color=node_colors, edge_color="gray")

    # Añadir las etiquetas de los nodos manualmente
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="darkblue")
    nx.draw_networkx_labels(G, pos, labels={provincia: provincia}, font_size=12, font_weight="bold", font_color="white")

    plt.title(f"Grafo de distribucion de estudiantes en {provincia}", fontsize=20)
    st.pyplot(plt)

# Llamar a la función para dibujar el grafo con la provincia seleccionada
update_graph(provincia)

import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def add_sidebar():
  st.sidebar.header("Mesures cellulaires 🦠")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Rayon (moyenne)", "radius_mean"),
        ("Texture (moyenne)", "texture_mean"),
        ("Périmètre (moyenne)", "perimeter_mean"),
        ("Surface (moyenne)", "area_mean"),
        ("Lissage (moyenne)", "smoothness_mean"),
        ("Compacité (moyenne)", "compactness_mean"),
        ("Concavité (moyenne)", "concavity_mean"),
        ("Points Concaves (moyenne)", "concave points_mean"),
        ("Symétrie (moyenne)", "symmetry_mean"),
        ("Dimension Fractale (moyenne)", "fractal_dimension_mean"),
        ("Rayon (erreur standard)", "radius_se"),
        ("Texture (erreur standard)", "texture_se"),
        ("Périmètre (erreur standard)", "perimeter_se"),
        ("Surface (erreur standard)", "area_se"),
        ("Lissage (erreur standard)", "smoothness_se"),
        ("Compacité (erreur standard)", "compactness_se"),
        ("Concavité (erreur standard)", "concavity_se"),
        ("Points Concaves (erreur standard)", "concave points_se"),
        ("Symétrie (erreur standard)", "symmetry_se"),
        ("Dimension Fractale (erreur standard)", "fractal_dimension_se"),
        ("Rayon (pire)", "radius_worst"),
        ("Texture (pire)", "texture_worst"),
        ("Périmètre (pire)", "perimeter_worst"),
        ("Surface (pire)", "area_worst"),
        ("Lissage (pire)", "smoothness_worst"),
        ("Compacité (pire)", "compactness_worst"),
        ("Concavité (pire)", "concavity_worst"),
        ("Points Concaves (pire)", "concave points_worst"),
        ("Symétrie (pire)", "symmetry_worst"),
        ("Dimension Fractale (pire)", "fractal_dimension_worst")
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Rayon', 'Texture', 'Périmètre', 'Surface', 
                'Lissage', 'Compacité', 
                'Concavité', 'Points Concaves',
                'Symétrie', 'Dimension Fractale']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Valeur moyenne'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Erreur standard'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Pire Valeur'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Prédiction des groupes de cellules 🔬")
  st.write("Le groupe de cellules est :")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Bénignes</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malignes</span>", unsafe_allow_html=True)
    
  
  st.write("Probabilité d'être bénin :", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probabilité d'être maligne :", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("Cette application peut aider les professionnels de la santé à établir un diagnostic, mais ne doit pas être utilisée comme substitut à un diagnostic professionnel.")



def main():
  st.set_page_config(
    page_title="Prédicteur du cancer du sein",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("Prédicteur du cancer du sein 👩‍⚕️")

    st.write("Connectez cette application à votre laboratoire de cytologie pour diagnostiquer le cancer du sein à partir des échantillons de tissu.")
    st.write("Cette application prédit si une masse mammaire est bénigne ou maligne à l'aide d'un modèle d'apprentissage automatique.")
    st.write("Vous pouvez également mettre à jour les mesures manuellement via les curseurs de la barre latérale.")

  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()
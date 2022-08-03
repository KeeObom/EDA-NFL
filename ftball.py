import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# Function to get the base and decode the image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Function to use CSS and set the background to be png_file
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('bkg-logo-1.jpg')


st.title('NFL Football Stats (Rushing) Explorer')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data (focusing on Rushing)!
\nMade by KeeObom
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib and seaborn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/years/2021/rushing.htm)
""")
# Sidebar user selection
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990, 2022))))

# web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2021/rushing.htm
@st.cache
def load_data(year):
	url = "https://www.pro-football-reference.com/years/"+str(year)+"/rushing.htm"
	html = pd.read_html(url, header=1)
	df = html[0]
	raw = df.drop(df[df.Age == 'Age'].index) # Deletes index of age string appearing in the age column	
	raw = raw.fillna(0)
	playerstats = raw.drop(['Rk'], axis=1)
	return playerstats

playerstats = load_data(selected_year)

# sidebar -- Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# sidebar -- position selection
unique_pos = ['RB', 'QB', 'WR', 'FB', 'TE']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

df_selected_team = playerstats[(playerstats.Tm.isin(selected_team))&(playerstats.Pos.isin(selected_pos))]

st.header("Display Player Stats of Selected Team(s)")
st.write('Data Dimension:' + str(df_selected_team.shape[0])+' rows and '+str(df_selected_team.shape[1])+' columns.')
st.dataframe(df_selected_team)

# Download NFL player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
	csv = df.to_csv(index=False)
	b65 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
	href = f'<a href="data:file/csv;base64,{b65}" download="playerstats.csv">Download CSV File'
	return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)
# Remove intercorrelation warning error
st.set_option('deprecation.showPyplotGlobalUse', False)

# Heatmap
if st.button('Intercorrelation Heatmap'):
	st.header('Intercorrelation Matrix Heatmap')
	df_selected_team.to_csv('output.csv',index=False)
	df = pd.read_csv('output.csv')

	corr = df.corr()
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True
	with sns.axes_style("white"):
		f, ax = plt.subplots(figsize=(7, 5))
		ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
	st.pyplot()

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Fantasy Stats Explorer')

st.markdown("""
This app allows for the comparison and analysis of NBA players for fantasy
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")





st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2022))))

# Sort stats
pStats = ['Age','MP','FG','FGA','FG%', '3P','3PA','2P','2PA','eFG%', 'FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','Fantasy Points']
selected_stat = st.sidebar.selectbox('Sort by Stat', pStats)

st.sidebar.header('Fantasy Basketball Values')
pts = st.sidebar.number_input('PTS',1)
misses = st.sidebar.number_input('FGM',0.45)
assists = st.sidebar.number_input('AST',1.5)
rebounds = st.sidebar.number_input('TRB', 1.2)
steals = st.sidebar.number_input('STL',3)
blocks = st.sidebar.number_input('BLK',3)
turnovers = st.sidebar.number_input('TOV',1)

# Web scraping of NBA player stats
@st.cache
# Data filtering wity pandas
def load_data(year, sorted, total):
    url = ""
    if total:
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_totals.html"
    else:
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    
    # - (df['FGA']-df['FG'])*0.45 +df['TRB']*1.2 + df['AST']*1.5 + df['STL']*3 + df['BLK']*3 -df['TO']
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    playerstats['Fantasy Points'] = playerstats['PTS']
    for index, row in playerstats.iterrows():
        row[29] = float(row[28])*pts - (float(row[8]) - float(row[7]))*misses + rebounds * float(row[22]) + assists*float(row[23]) + steals*float(row[24]) + blocks*float(row[25]) - float(row[26])*turnovers

    playerstats[sorted] = pd.to_numeric(playerstats[sorted])
    playerstats = playerstats.sort_values(by = sorted, ascending = False)
    return playerstats

playerstats = load_data(selected_year,selected_stat, False)
playerTotal = load_data(selected_year, selected_stat, True)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)







# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]


st.header('Per Game Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)


# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

st.header('Total Player Stats of Selected Team(s)')
df_selected_total = playerTotal[playerTotal.Tm.isin(selected_team) & (playerTotal.Pos.isin(selected_pos))]
st.dataframe(df_selected_total)
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
import sqlite3
import pandas as pd
import streamlit as st

conn = sqlite3.connect('data_bondapp.db')
cursor = conn.cursor()
cursor.execute('select * from `bankattr`')
df = pd.DataFrame(cursor.fetchall())

st.dataframe(df, 1000, 500)


if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
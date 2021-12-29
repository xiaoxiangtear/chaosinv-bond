import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import datetime as dt

## >>>>> 初始设置 <<<<<
st.set_page_config(
     page_title="混沌天成投顾业务管理系统",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)
banks_lt = ['QA', 'LT', 'YT2']

## >>>>> 获得原始数据 <<<<<
@st.cache
def initializeData():
    conn = sqlite3.connect('data_bondapp.db')
    DEAL = pd.read_sql('select * from `bonddeal`', con=conn)
    ATTR = pd.read_sql('select * from `bankattr`', con=conn)
    BANK = pd.read_sql('select * from `bankadmin`', con=conn)
    BONDINFO = pd.read_sql('select * from `bondinfo`', con=conn)
    CNBD = pd.read_sql('select * from `cnbd`', con=conn)

    DEAL = DEAL[DEAL['账户'].isin(banks_lt)]
    ATTR = ATTR[ATTR['账户'].isin(banks_lt)]
    BANK = BANK[BANK['账户'].isin(banks_lt)]

    return DEAL, ATTR, BANK, BONDINFO, CNBD
DEAL, ATTR, BANK, BONDINFO, CNBD = initializeData()

## >>>>> 辅助函数 <<<<<<
## 数据转换
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

## 获得指定日的业绩数据
@st.cache
def getData_attr(data_df, cal_date):
    df = data_df.copy()
    ib = pd.to_datetime(df['日期']) <= pd.to_datetime(cal_date)
    df = df[ib].sort_values(by='日期', ascending=True)
    last_date = df['日期'].tolist()[-1]
    ib2 = df['日期'] == last_date
    attr_df = df[ib2]
    return attr_df

## 计算指定日的持仓数据
@st.cache
def calPostion(deal_df, cal_date=None):

    if cal_date is None:
        cal_date = dt.datetime.today()

    df = deal_df.copy()
    # 增加债券简称
    bondinfo_df = BONDINFO.set_index('债券代码')
    df['债券简称'] = [bondinfo_df.at[x, '债券简称'] for x in df['债券代码']]
    ib = pd.to_datetime(df['结算日']) <= pd.to_datetime(cal_date)
    df = df[ib]

    buy_df = df[df['交易方向'] == '买入']
    buy_df.index = buy_df['成交编号']
    sell_df = df[df['交易方向'] == '卖出']
    n = sell_df.shape[0]
    index_lt = sell_df.index
    for i in range(n):
        sellfrom = sell_df.at[index_lt[i], '卖出对应']
        volume = sell_df.at[index_lt[i], '成交券面']
        dealID = sell_df.at[index_lt[i], '成交编号']
        sellfrom_lt = [x.split(':') for x in sellfrom.split(',')]
        amount = 0
        for sell in sellfrom_lt:
            sell_id = sell[0]
            sell_amount = float(sell[1])
            # 删去对应的buy记录
            buy_df.at[sell_id, '成交券面'] -= sell_amount
            amount += sell_amount
        if volume != amount: print(f'{dealID}卖出对应有问题！')

    # 删除成交券面为0的记录
    ib = buy_df['成交券面'] != 0
    buy_df = buy_df[ib]

    # 还需删掉已到期的
    buy_df['到期日'] = buy_df['债券代码'].apply(lambda x: pd.to_datetime(bondinfo_df.at[x, '到期日']).date())
    buy_df = buy_df.sort_values(by='到期日', ascending=True)
    buy_df = buy_df[buy_df['到期日'] >= dt.datetime.today().date()]
    buy_df = buy_df.drop(columns=['对手方', '备注'])

    # 根据债券代码合并持仓
    bondvalue_df = CNBD.copy()
    position_df = pd.DataFrame([])
    banks_lt = list(df['账户'].unique())
    for bank in banks_lt:
        ib_bank = buy_df['账户'] == bank
        bank_df = buy_df[ib_bank]
        lt = []
        for bond in bank_df['债券代码'].unique():
            ib_bond = bank_df['债券代码'] == bond
            bond_volume = bank_df[ib_bond]['成交券面'].sum()
            bond_ytm = (bank_df[ib_bond]['成交收益率'] * bank_df[ib_bond]['成交券面']).sum() / bank_df[ib_bond]['成交券面'].sum()
            bond_price1 = (bank_df[ib_bond]['成交净价'] * bank_df[ib_bond]['成交券面']).sum() / bank_df[ib_bond]['成交券面'].sum()
            bond_price2 = (bank_df[ib_bond]['成交全价'] * bank_df[ib_bond]['成交券面']).sum() / bank_df[ib_bond]['成交券面'].sum()
            bond_name = bondinfo_df.at[bond, '债券简称']
            bond_coupon = bondinfo_df.at[bond, '票面利率']
            bond_coupondate = bondinfo_df.at[bond, '起息日']
            bond_maturitydate = bondinfo_df.at[bond, '到期日']
            bond_term = (pd.to_datetime(bond_maturitydate) - pd.to_datetime(cal_date)).days / 365
            ib_cnbd = (bondvalue_df['债券代码'] == bond) & (pd.to_datetime(bondvalue_df['日期']) <= pd.to_datetime(cal_date))
            temp_df = bondvalue_df[ib_cnbd].sort_values(by='日期', ascending=False).reset_index()
            bond_ytm_now = temp_df['估价收益率'][0]
            bond_price1_now = temp_df['估价净价'][0]
            bond_profitloss = (bond_price1_now - bond_price1) / 100 * bond_volume
            lt.append([bank,
                       bond,
                       bond_name,
                       bond_volume,
                       bond_coupon,
                       bond_coupondate,
                       pd.to_datetime(bond_maturitydate).date(),
                       round(bond_term, 2),
                       round(bond_ytm, 4),
                       round(bond_price1, 4),
                       round(bond_price2, 4),
                       round(bond_ytm_now, 4),
                       round(bond_profitloss, 2)])
        pos_df = pd.DataFrame(lt, columns=['账户', '债券代码', '债券简称', '持仓券面', '票面利率', '起息日', '到期日', '剩余期限', '成本收益率', '成本净价',
                                           '成本全价', '当前收益率', '净价浮盈(万元)'])
        position_df = pd.concat([position_df, pos_df], axis=0, sort=False)
    return buy_df, position_df

## >>>>> 主体 <<<<<
## 边侧栏
with st.sidebar:
    st.image('logo.png')
    date = st.date_input('选择日期：')
    bank = st.selectbox('选择账户：',banks_lt)
    buy_df, position_df = calPostion(DEAL, date)
    st.download_button(
        label="导出成交数据",
        data=convert_df(DEAL),
        file_name='deal.csv',
        mime='text/csv',
    )
    st.download_button(
        label="导出持仓数据(分笔)",
        data=convert_df(buy_df),
        file_name='position(分笔).csv',
        mime='text/csv',
    )
    st.download_button(
        label="导出持仓数据(分券)",
        data=convert_df(position_df),
        file_name='position(分券).csv',
        mime='text/csv',
    )
    st.download_button(
        label="导出业绩数据",
        data=convert_df(pd.DataFrame(np.random.random([100, 3]))),
        file_name='attr.csv',
        mime='text/csv',
    )
    attr_df = getData_attr(ATTR, date)
    last_date = attr_df['日期'].tolist()[-1]
    st.write('更新日：', last_date)

## 主界面
# 1、账户基本情况
attr_df = getData_attr(ATTR, date)
container1 = st.container()
container1.subheader('❉ 账户基本情况')
show_df1 = attr_df.set_index('账户')[['库存券面','库存全价','账户杠杆','账户久期1','账户久期2','账户DV01']]
bank_df = BANK.copy()[['账户','账户简称', '账户全称', '本金', '到账日']].set_index('账户')
show_df1 = pd.concat([show_df1, bank_df], axis=1, sort=False, join='inner')
show_df1['库存券面'] = show_df1['库存券面'].apply(lambda x: int(x))
show_df1['库存全价'] = show_df1['库存全价'].apply(lambda x: int(x))
show_df1['账户杠杆'] = show_df1['账户杠杆'].apply(lambda x: f'{x:.2%}')
show_df1['账户久期1'] = show_df1['账户久期1'].apply(lambda x: f'{x:.2f}')
show_df1['账户久期2'] = show_df1['账户久期2'].apply(lambda x: f'{x:.2f}')
show_df1['账户DV01'] = show_df1['账户DV01'].apply(lambda x: int(x))
container1.dataframe(data=show_df1, width=1200)

# 2、账户业绩情况
container2 = st.container()
container2.subheader('❉ 账户业绩情况')
show_df2 = attr_df.set_index('账户')[['净值','期间收益率','年化收益率','年化波动率','最大回撤率','收益回撤比', '夏普比率']]
show_df2['期间收益率'] = show_df2['期间收益率'].apply(lambda x: f'{x:.2%}')
show_df2['年化收益率'] = show_df2['年化收益率'].apply(lambda x: f'{x:.2%}')
show_df2['年化波动率'] = show_df2['年化波动率'].apply(lambda x: f'{x:.2%}')
show_df2['最大回撤率'] = show_df2['最大回撤率'].apply(lambda x: f'{x:.2%}')
container2.dataframe(data=show_df2, width=790)
nav_df = ATTR[['日期','账户','净值']]
nav_df = nav_df.pivot_table(values='净值', index='日期', columns='账户').sort_index(ascending=True)
fig = px.line(nav_df, labels=dict(value='净值曲线',index='日期',variable='账户'))
container2.plotly_chart(fig, use_container_width=True)

# 3、账户持仓情况
buy_df, position_df = calPostion(DEAL, date)
container3 = st.container()
container3.subheader('❉ 账户持仓情况')
container3.markdown('- 分笔：')
show_df3 = buy_df[['账户','交易方向','债券代码','债券简称','剩余券面','成交收益率','成交净价','成交全价','结算日']]
show_df3 = show_df3[show_df3['账户']==bank]
show_df3['剩余券面'] = show_df3['剩余券面'].apply(lambda x:int(x))
container3.dataframe(data=show_df3)
container3.markdown('- 分券：')
show_df4 = position_df[['账户','债券代码','债券简称','持仓券面','票面利率','到期日','剩余期限','成本收益率','成本净价','成本全价','当前收益率','净价浮盈(万元)']].reset_index(drop=True)
show_df4 = show_df4[show_df4['账户']==bank]
show_df4['持仓券面'] = show_df4['持仓券面'].apply(lambda x:int(x))
container3.dataframe(data=show_df4)

# 4、净值分析
container4 = st.container()
container4.subheader('❉ 净值分析')
col1, col2 = container4.columns(2)
nav_df2 = ATTR.copy()
ib = nav_df2['账户'] == bank
nav_df2 = nav_df2[ib].set_index('日期')[['净值','账户DV01']].dropna()
fig1 = px.line(nav_df2['净值'], labels=dict(value=bank,variable=''))
col1.plotly_chart(fig1, use_container_width=True)
fig2 = px.line(nav_df2['账户DV01'], labels=dict(value=bank,variable=''))
col2.plotly_chart(fig2, use_container_width=True)

# 5、业绩分析
container5 = st.container()
container5.subheader('❉ 业绩分析')
show_df5 = attr_df.set_index('账户')[['库存净价浮盈','总收益全价法','库存全价浮盈','卖断全价收益','卖断票息兑付','到期收益','正回购利息','逆回购利息']]
show_df5 = show_df5.loc[bank, :].astype('int')
container5.dataframe(show_df5.to_frame().T, width=1000)
fig3 = px.bar(show_df5[['总收益全价法','库存全价浮盈','卖断全价收益','卖断票息兑付','到期收益','正回购利息','逆回购利息']],labels=dict(value='收益分解',variable='', index=''))
container5.plotly_chart(fig3, use_container_width=False)

# 6、持仓分析
container6 = st.container()
container6.subheader('❉ 持仓分析')
col1, col2 = container6.columns(2)
show_df6 = attr_df.set_index('账户')
show_df6 = show_df6.loc[bank, :]
# 品种分布
dist_df1 = pd.DataFrame(
    [
        ['国债', show_df6['国债']],
        ['国开债', show_df6['国开债']],
        ['进出债', show_df6['进出债']],
        ['农发债', show_df6['农发债']],
        ['同业存单', show_df6['同业存单']]
    ], columns=['品种', '占比'])
fig1 = px.pie(data_frame=dist_df1,values="占比", names="品种")
col1.plotly_chart(fig1, use_container_width=True)

# 期限分布
dist_df2 = pd.DataFrame(
    [
        ['3m-', show_df6['小于3月']],
        ['3m~6m', show_df6['由3至6月']],
        ['6m~1y', show_df6['由6月至1年']],
        ['1y~3y', show_df6['由1至3年']],
        ['3y~5y', show_df6['由3至5年']],
        ['5y~7y', show_df6['由5至7年']],
        ['7y~10y', show_df6['由7至10年']],
        ['10y+', show_df6['大于10年']]
    ], columns=['期限', '占比'])
fig2 = px.pie(data_frame=dist_df2,values="占比",names="期限")
col2.plotly_chart(fig2, use_container_width=True)

## >>>>> css设置 <<<<<
hide_menu_style = """
        <style>
            #MainMenu {visibility: hidden;}
            .css-uc7ato {margin-top: 10px;width: 304px;background-color: #f3b352;border:none;}
            .css-qrbaxs {margin-top: 1rem;}
            h3 {margin-bottom: 1rem;margin-top: 1rem;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)


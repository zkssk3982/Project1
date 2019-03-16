from stock.stock_model import StockModel
from selenium import webdriver
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    driver = webdriver.Chrome('chromedriver')
    url = 'https://bitcoincharts.com/charts/korbitKRW#rg730ztgSzm1g10zm2g25zv'
    driver.get(url)
    xpath = """//*[@id="content_chart"]/div/div[2]/a"""
    variable = driver.find_element_by_xpath(xpath)
    driver.execute_script("return arguments[0].scrollIntoView();", variable)
    variable.click()
    #그 메뉴가 보일 때 까지 스크롤해서 내려가서 클릭하라
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    #print(soup.prettify())

    table = soup.find_all('table', id='chart_table')
    print(table)

    df = pd.read_html(str(table))
    df.Capacity = df.Capacity.str.replace(r"\[.*\]", "")
    bitcoin = df[0]
    print(bitcoin.all())

    bitcoin['Close'].plot(figsize=(12, 6), grid=True)
    df = pd.DataFrame({'ds': bitcoin['Timestamp]'], 'y': bitcoin['Close']})
    print(df)
    prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)
    prophet.fit(df)
    future = prophet.make_future_dataframe(perods=30)
    forecast = prophet.predict(future)
    prophet.plot(forecast)



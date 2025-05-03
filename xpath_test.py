from selenium.webdriver import Edge
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
# # pip install selenium webdriver-manager
from webdriver_manager.microsoft import EdgeChromiumDriverManager # Import trình quản lý cho Edge
import pandas as pd
import time
from selenium.webdriver.common.keys import Keys

url = "https://www.worldometers.info/coronavirus/#countries"

edge_options = Options()
edge_options.add_experimental_option("detach", True)

driver = Edge(service=Service(EdgeChromiumDriverManager().install()),
              options=edge_options)
driver.get(url)
time.sleep(2) 
driver.maximize_window()

# show all columns
show_columns_xpath = '//*[@id="main_table_controls"]/ul/div/button'
show_columns_button = driver.find_element("xpath", show_columns_xpath)
show_columns_button.click()

# Click all features
# features_xpath = f'/html/body/div[3]/div[3]/div/div[6]/ul/div/ul/li[2]/div/label/input'
features_xpath = f'/html/body/div[3]/div[3]/div/div[6]/ul/div/ul/li'
features_button = driver.find_elements("xpath", features_xpath)
for feature in features_button:
    if (f'{feature}//input[not(@checked])'):
        feature.click()
        time.sleep(0.5)

for feature in features_button:
    if (f'{feature}//input[not(@checked])'):
        feature.click()
        time.sleep(0.5)

# <input id="column_2" checked="" type="checkbox" class="toggle-vis" data-column="2">
# <input id="column_3" type="checkbox" class="toggle-vis" data-column="3">

xpath_col_query = '//table[@id="main_table_countries_today"]//thead/tr/th'
columns = driver.find_elements("xpath", xpath_col_query)

columns_name = [column.text for column in columns]
df = pd.DataFrame(columns=columns_name)

print(df.head())



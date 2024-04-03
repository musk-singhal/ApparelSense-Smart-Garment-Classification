from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import requests
import pandas as pd
import requests
import os


def capture_data(page_range=100, dataset_path="img_dataset"):
    def download_image(url, save_path, image_name):
        response = requests.get(url)

        if response.status_code == 200:
            save_directory = os.path.join(dataset_path, save_path)

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            with open(os.path.join(save_directory, image_name), 'wb') as file:
                file.write(response.content)

    driver = webdriver.Chrome()

    driver.get("https://www.myntra.com/")

    # Wait for page to load
    time.sleep(3)
    count = 0
    data = {'product_name': [], 'product_brand': [], 'product_url': [], 'product_price': [], 'product_img_url': [],
            'product_category': []}
    # Find the search input element and input text
    search_input = driver.find_element(By.CLASS_NAME, 'desktop-searchBar')
    for title in ['men full sleeves t-shirts', 'men half sleeves t-shirts', 'women full sleeves t-shirts',
                  'women half sleeves tshirt']:
        print(f"Start capturing {title} images...")
        count = 0
        search_input = driver.find_element(By.CLASS_NAME, 'desktop-searchBar')

        search_input.send_keys(title)

        # Find the search button and click on it
        search_button = driver.find_element(By.CLASS_NAME, 'desktop-submit')
        search_button.click()

        # Wait for search results to load

        for page in range(page_range):
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollTo( document.body.scrollHeight, 0);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            # Find all the product elements
            product_elements = driver.find_element(By.CLASS_NAME, 'results-base')
            product_elements = product_elements.find_elements(By.CLASS_NAME, "product-base")

            # Print details of each product
            for product in product_elements:
                try:
                    product_url = product.find_elements(By.TAG_NAME, "a")
                    product_url = product_url[0].get_attribute('href')
                except:
                    product_url = ""
                try:
                    img_tag = product.find_elements(By.TAG_NAME, "img")
                    img_url = img_tag[0].get_attribute('src')
                    download_image(img_url, title, str(count) + '.png')
                    count += 1
                except:
                    img_url = ""
                try:
                    brand = product.find_elements(By.CLASS_NAME, 'product-brand')[0].text
                except:
                    brand = "N/A"
                try:
                    product_name = product.find_elements(By.CLASS_NAME, 'product-product')[0].text
                except:
                    product_name = "N/A"
                try:
                    price = product.find_elements(By.CLASS_NAME, 'product-discountedPrice')[0].text
                except:
                    price = 'Rs. -1'
                data['product_name'].append(product_name)
                data['product_brand'].append(brand)
                data['product_url'].append(product_url)
                data['product_price'].append(price)
                data['product_img_url'].append(img_url)
                data['product_category'].append(title)
                if count % 100 == 0:
                    print(f"Till Now total captured images:- {count} for category:- {title}")
            if page + 1 == page_range:
                break
            next_page_link = driver.find_element('xpath', "//li[@class='pagination-next']/a")
            driver.execute_script("arguments[0].click();", next_page_link)
            time.sleep(5)
    df = pd.DataFrame(data)
    df.to_csv('dataset.csv')

    # Close the WebDriver
    driver.quit()

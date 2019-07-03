import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


driver_path = "C:\\tensorflow1\\chromedriver.exe"

chr_driver = webdriver.Chrome(driver_path)

chr_driver.get("https://www.google.co.kr")


chr_driver.close()
chr_driver.quit()
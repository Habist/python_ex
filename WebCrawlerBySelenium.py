import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def getText(drive: webdriver.chrome, name: str) :
    return drive.find_element_by_class_name(name).text

driver_path = "C:\\tensorflow1\\chromedriver.exe"

chr_driver = webdriver.Chrome(driver_path)
chr_driver.get("https://www.naver.com")
chr_driver.find_element_by_name("query").send_keys("날씨")
chr_driver.find_element_by_id("search_btn").click()

WebDriverWait(chr_driver, 100).until(EC.presence_of_all_elements_located)
today = chr_driver.find_elements_by_class_name("menu")[4].text

ment = today + "의 '" + getText(chr_driver, "btn_select") + "' 기온은 " + getText(chr_driver,"todaytemp") + "도 입니다."
print(ment)


# for idx, val in enumerate(elements):
#    print(idx)
#    print(val.text)

# chr_driver.close()
# chr_driver.quit()




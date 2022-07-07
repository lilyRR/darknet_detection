from selenium.webdriver import Chrome
import os
from selenium import webdriver
import time
chromeOptions = webdriver.ChromeOptions()
prefs = {"download.default_directory": "D:\\binvis"}
chromeOptions.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(chrome_options=chromeOptions)
folder = 'D:\\pacap\\result'
for root, dirs, files in os.walk(folder):
    for file in files:
#for x in file_name_list:

        driver.get('http://www.binvis.io/#/')
        time.sleep(0.5)
        driver.find_element_by_xpath('//*[@id="filebutton"]').click()
        time.sleep(1)
    #path='E:\\TorPcaps\\Pcaps\\tor\\AUDIO_spotifygateway.pcap'

        file_input = driver.find_element_by_id("fileinput")
        file_input.send_keys(root+'/'+file)

        time.sleep(0.15)
        driver.find_element_by_xpath('//i[@class="fa fa-lg fa-camera-retro"]').click()
        time.sleep(0.15)
        driver.find_element_by_xpath('//button[@class="btn btn-primary"]').click()
        print(file+"下载完成")
        time.sleep(0.01)

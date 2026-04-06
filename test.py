from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
#百度搜索内容
driver =webdriver.Chrome()#初始化浏览器
#⽤get打开百度⻚⾯
driver.get("http://www.baidu.com")
#找到百度的输⼊框，并输⼊“⼤数据”
driver.find_element(By.ID,'kw').send_keys('⼤数据应⽤')
sleep(5)
#点击搜索按钮
driver.find_element(By.ID,'su').click()
sleep(5)
content= driver.find_element(By.ID,'content_left').text
print(content)
driver.quit()
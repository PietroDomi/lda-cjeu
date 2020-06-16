from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
driver.get("https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:62014CJ0371")
with open(str(driver.title)+".html","w",encoding='utf-8') as file:
    file.write(driver.page_source)
print(driver.title)
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()
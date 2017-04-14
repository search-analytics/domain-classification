
# coding: utf-8

# ## Web of Science Scaper for Foundry Project
# 
# This notebook has selenium script for scraping CSV files from  Web of Science
# 
# ### Requirements
# 1. pip install selenium
# 2. Firefox must be there on the machine which runs this
# 3. Machine should be on JPL network, so that we can access Web of Science without Authentication hassles
# 
# 

# In[16]:

# Requirements :
#  1. pip install selenium
#
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import os
import time

cs_labels = ["Computer Science, Software Engineering", "Computer Science, Cybernetics", "Computer Science, Hardware & Architecture", 
"Computer Science, Information Systems", "Computer Science, Theory & Methods", "Computer Science, Artificial Intelligence", 
"Computer Science, Interdisciplinary Applications"]

wos_cats = []

with open(os.path.join("categories", "WoS-categories"), "r") as cats:
    for x in cats.readlines():
        if x.replace("\n","") in cs_labels: 
            wos_cats.append(x)    

    #************************************************************************************
    #IMPORTANT: Must update this URL if running the script for first time in awhile
    #************************************************************************************
    url = "http://apps.webofknowledge.com/WOS_AdvancedSearch_input.do?SID=2DTDPX6n7D51h8m4kEH&product=WOS&search_mode=AdvancedSearch"
    delay = 5 # seconds
    downloads_dir = os.path.join(os.getcwd(), 'data')
    log_file = open('logs.txt', 'wb', 1)

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)
        
    def log(msg, stdout=True):
        msg = time.strftime("%Y/%m/%d-%H:%M:%S :: ") + msg
        log_file.write(msg)
        log_file.write("\n")
        if stdout:
            print(msg)

    # log("saving files to %s" % downloads_dir)
    # Configure firefox to auto save downloads at a specific directory
    # fp = webdriver.ChromeProfile()
    # fp.set_preference("browser.download.folderList",2) 
    # fp.set_preference("browser.download.manager.showWhenStarting", False)
    # fp.set_preference("browser.download.dir", downloads_dir)
    # fp.set_preference("browser.helperApps.neverAsk.saveToDisk","text/csv,text/plain")

    driver = webdriver.Chrome()
    log("Browser launched with URL: %s" % url)

    driver.get(url)
    time.sleep(3)

    driver.find_element_by_xpath('//*[@title="Advanced Search"]').click()

for idx, cat in enumerate(wos_cats):

    query = "WC=" + str(cat)

    if idx > 0:
        driver.find_element_by_xpath('//*[@title="Back to Search"]').click()
        driver.find_element_by_xpath('//*[@name="selsets"]').click()
        driver.find_element_by_xpath('//*[@title="Delete selected sets"]').click()


    time.sleep(1)
    log("Searching %s" % query)

    searchbox = driver.find_element_by_id('value(input1)')
    searchbox.clear()
    searchbox.send_keys(query)

    driver.find_element_by_xpath('//*[@title="Search"]').click()
    driver.find_element_by_id('set_1_div').click()
    # searchbox.send_keys(Keys.RETURN)

    # select maximum entries possible per page,
    Select(driver.find_element_by_id('selectPageSize_.bottom')).select_by_value("50")

    # 
    # ## Resume the crawl
    # Look into the logs => get the page number => then go to that page in firefox browser (change URL param and hit Enter) connected to selenium and then execute the below cell

    # In[13]:

    delay = 5
    page_limit = 30
    start_page = 29
    page = 0


    has_next = True
    while page < start_page + page_limit:

        while page <= start_page: 
            next_click = driver.find_element_by_xpath("//*[@alt='Next Page']")
            log("Going to next page", stdout=False)
            next_click.click()
            page += 1

        page += 1

        log("URL = %s " % driver.current_url, stdout=False)
        
        # select other format
        Select(driver.find_element_by_id('saveToMenu')).select_by_value("other")

        # time.sleep(2)
        # in the following dialogue select fields, including Abstract
        Select(driver.find_element_by_id('bib_fields')).select_by_value("PMID USAGEIND AUTHORSIDENTIFIERS ACCESSION_NUM FUNDING SUBJECT_CATEGORY JCR_CATEGORY LANG IDS PAGEC SABBR CITREFC ISSN PUBINFO KEYWORDS CITTIMES ADDRS CONFERENCE_SPONSORS DOCTYPE CITREF ABSTRACT CONFERENCE_INFO SOURCE TITLE AUTHORS  ")

        # select the format
        Select(driver.find_element_by_id('saveOptions')).select_by_value("fieldtagged")

        # send these selections
        driver.find_element_by_xpath("//*[@class='quickoutput-action']//*[@name='email']").click()

        log("Waiting for %d secs" % delay, stdout=False)
        time.sleep(delay)
        #close the dialogue/popup
        driver.find_element_by_xpath("//a[@class='quickoutput-cancel-action']").click()
        
        next_click = driver.find_element_by_xpath("//*[@alt='Next Page']")
        has_next = next_click

        if has_next:
            log("Going to next page", stdout=False)
            next_click.click()
            # find next button and click

    log("Reached the end.. URL=%s" % driver.current_url)


    # In[ ]:

log_file.close()
driver.close()
driver.quit()
    # Done


# In[ ]:

import time
time.strftime("%Y/%m/%d-%H:%M:%S")


# In[ ]:




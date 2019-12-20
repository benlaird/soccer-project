import time

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options

html = None


delay = 2  # seconds

chrome_options = Options()
# chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)


def get_all_games_for_season_in_csv(season_str):
    driver.get(url)

    html = driver.page_source
    time.sleep(delay)

    action = ActionChains(driver)

    if html:
        # soup = BeautifulSoup(html, 'html.parser')
        # span[contains(text(), 'Assign Rate')]
        driver.execute_script("window.scrollTo(0, 500)")
        shareMenu = driver.find_element_by_xpath("//span[contains(text(),'Share')]")
        print(shareMenu.get_attribute('innerHTML'))
        parent = shareMenu.find_element_by_xpath("./..")
        print(parent.get_attribute('innerHTML'))

        action.move_to_element(shareMenu).perform()

        # secondLevelMenu = parent.find_element_by_partial_link_text('CSV')
        secondLevelMenu = parent.find_element_by_xpath("//button[contains(text(), 'Get table as CSV')]")
        action.move_to_element(secondLevelMenu).perform()
        secondLevelMenu.click()
        input("Press any key to continue")
        #  < pre id = "csv_sched_ks_1889_1"
        csvTag = driver.find_element_by_id("csv_sched_ks_1889_1")
        return csvTag.text

def close_selenium():
    # shareMenu.clear()
    driver.close()

def get_all_seasons_csv():
    seasons = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    url_prefix = "https://fbref.com/en/comps/9/1889/schedule/2018-2019-Premier-League-Fixtures"
    url_postfix = "-Premier-League-Fixtures"

    for s in seasons:
        url = url_prefix + f("{s}-{s+1}") + url_postfix
        filename = "{s}-{s+1}.csv"
        f = open(filename, "w+")
        csv_text = get_all_games_for_season_in_csv(url)
        f.write(csv_text)
        f.close()


import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoAlertPresentException,
)
from selenium.webdriver.common.action_chains import ActionChains

__author__ = "Åsmund Frantzen Skomedal"
__copyright__ = "Åsmund Frantzen Skomedal"
__license__ = "MIT"

# URL = r"https://politiskagenda.kristiansand.kommune.no/?request.kriterie.udvalgId=355a0ec6-ac1f-4e76-8ad4-3ab898841838&request.kriterie.moedeDato=2024"
URL = r"https://politiskagenda.kristiansand.kommune.no/"


def main(url: str) -> None:
    """Use beatifulsoup to scrape kommune data from the given url."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # utvalg = soup.find(id='udvalg')
        # search_button = soup.find(id='searchButton')
        # resultater_div = soup.find(id='resultater')

        # Initialize Selenium WebDriver (make sure to have the appropriate driver installed)
        driver = webdriver.Firefox()  # or webdriver.Firefox(), etc.
        driver.get(URL)  # Replace with the actual URL

        # Wait until the dropdown is present
        wait = WebDriverWait(driver, 10)
        dropdown = wait.until(
            EC.visibility_of_element_located((By.ID, "multidropdown"))
        )

        # # Print the list of options available in the dropdown
        # options = driver.execute_script("""
        #     var dropdown = document.getElementById('multidropdown');
        #     return Array.from(dropdown.children).map(child => child.textContent);
        # """)
        # print("Options available in the dropdown before insertion:")
        # for option in options:
        #     print(option)

        # Use ActionChains to interact with the dropdown
        actions = ActionChains(driver)
        actions.move_to_element(dropdown).click().perform()

        # Select the parent option 'Valgperioden 2023-2027'
        parent_option = wait.until(
            EC.visibility_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='dropdown-item' and text()='Valgperioden 2023-2027']",
                )
            )
        )
        actions.move_to_element(parent_option).click().perform()

        # Select the suboption 'Areal- og miljøutvalget'
        sub_option = wait.until(
            EC.visibility_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='dropdown-item' and text()='Areal- og miljøutvalget']",
                )
            )
        )
        actions.move_to_element(sub_option).click().perform()

        # Enable and click the search button
        search_button = driver.find_element(By.ID, "searchButton")
        driver.execute_script("arguments[0].removeAttribute('disabled')", search_button)
        search_button.click()

        # Handle unexpected alerts and wait for the results
        try:
            wait.until(EC.presence_of_element_located((By.ID, "resultater")))
        except UnexpectedAlertPresentException:
            try:
                alert = driver.switch_to.alert
                alert.dismiss()
            except NoAlertPresentException:
                pass
            wait.until(EC.presence_of_element_located((By.ID, "resultater")))

        # Explicitly wait for the results to be visible
        wait.until(EC.visibility_of_element_located((By.ID, "resultater")))

        # Get the results
        resultater = driver.find_element(By.ID, "resultater")
        print(resultater.text)

        # Prompt the user for input to continue or exit
        user_input = input("Continue?").strip().lower()
        print("Exiting the program.")

        # Close the driver
        driver.quit()

    else:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")


if __name__ == "__main__":
    main(url=URL)

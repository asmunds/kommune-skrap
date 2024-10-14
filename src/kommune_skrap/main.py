import os

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

__author__ = "Åsmund Frantzen Skomedal"
__copyright__ = "Åsmund Frantzen Skomedal"
__license__ = "MIT"

# URL = r"https://politiskagenda.kristiansand.kommune.no/?request.kriterie.udvalgId=355a0ec6-ac1f-4e76-8ad4-3ab898841838&request.kriterie.moedeDato=2024"
URL = r"https://politiskagenda.kristiansand.kommune.no/"

DOWNLOAD_DIR = "../data/"  # Directory to save downloaded PDF files

KEYWORDS = ["dispensasjon", "søknad"]  # Keywords to search for in the scraped data


def download_pdf(url, download_dir):
    """Download a PDF file from the given URL to the specified directory.

    Args:
        url (str): URL to the PDF file.
        download_dir (str): Directory to save the PDF file.
    """
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(download_dir, url.split("/")[-1])
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")


def main(url: str) -> None:
    """Use beatifulsoup to scrape kommune data from the given url.

    Args:
        url (str): URL to scrape data from.
    """
    # Create download directory if it doesn't exist (check if parent directory exists)
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR, parents=False)

    response = requests.get(url)
    if response.status_code == 200:
        # Initialize Selenium WebDriver (make sure to have the appropriate driver installed)
        driver = webdriver.Firefox()
        driver.get(URL)

        # Wait until the dropdown is present
        wait = WebDriverWait(driver, 10)

        # Prompt the user for input to continue or exit
        user_input = input("\nMake relevant search, then press enter").strip().lower()

        # Get the results
        resultater = driver.find_element(By.ID, "resultater")

        # Find all links in the resultater table
        links = resultater.find_elements(By.TAG_NAME, "a")

        # Click on each link to enter a new web page
        for link in links:
            link.click()

            # Wait until the new page is loaded
            wait.until(lambda driver: driver.current_url != URL)
            wait.until(lambda driver: driver.find_element(By.TAG_NAME, "body"))
            # Wait until the number of links is greater than 2
            wait.until(lambda driver: len(driver.find_elements(By.TAG_NAME, "a")) > 2)

            page_links = driver.find_elements(By.TAG_NAME, "a")

            # Find and click all '+' signs to reveal hidden data
            plus_signs = driver.find_elements(
                By.CSS_SELECTOR, "span.glyphicon.glyphicon-plus[aria-hidden='true']"
            )
            for plus_sign in plus_signs:
                plus_sign.click()

                # Get the title of the section where this plus sign is located
                section_title = plus_sign.find_element(
                    By.XPATH,
                    "./ancestor::tr/td[2]/button/h2[contains(@class, 'overskrift')]",
                ).text
                if any(keyword in section_title.lower() for keyword in KEYWORDS):
                    # Get the new links that have appeared after clicking the plus sign
                    # Find links that are located in this row of the table
                    subpage_links = plus_sign.find_elements(
                        By.XPATH, "./ancestor::tr//a"
                    )
                    # Loop through the new links to find the one that contains the PDF
                    for new_link in subpage_links:
                        new_href = new_link.get_attribute("href")
                        if (
                            new_href
                            and new_href.endswith("Pdf=false")
                            and "Link til sak" in new_link.text
                        ):
                            print(f"Section Title: {section_title}")
                            new_link.click()
                            # Wait until the new page is loaded
                            wait.until(lambda driver: driver.current_url != URL)
                            # Switch to the new window/tab
                            driver.switch_to.window(driver.window_handles[-1])

                            # Get the URL of the PDF
                            pdf_url = driver.current_url

                            # Download the PDF
                            download_pdf(pdf_url, DOWNLOAD_DIR)

                            # Close the PDF window/tab and switch back to the original window/tab
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])

            # Add any additional scraping or processing logic here
            driver.back()  # Go back to the previous page to continue with the next link
            break

        # Close the driver
        driver.quit()

    else:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")


if __name__ == "__main__":
    main(url=URL)

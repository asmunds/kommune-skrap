import os

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

__author__ = "Åsmund Frantzen Skomedal"
__copyright__ = "Åsmund Frantzen Skomedal"
__license__ = "MIT"


URL = r"https://politiskagenda.kristiansand.kommune.no/"

DOWNLOAD_DIR = "./data/"  # Directory to save downloaded PDF files

KEYWORDS = ["dispensasjon"]  # Keywords to search for in the scraped data


def download_pdf(*, url, download_dir, new_filename: str):
    """Download a PDF file from the given URL to the specified directory.

    Args:
        url (str): URL to the PDF file.
        download_dir (str): Directory to save the PDF file.
    """
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(download_dir, new_filename + ".pdf")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {new_filename}.pdf")
    else:
        print(f"Failed to download: {new_filename}.pdf")


def identify_keyword(*, section_title, keywords):
    """Identify which keyword is present in the section title.

    Args:
        section_title (str): The title of the section.
        keywords (list): List of keywords to search for.

    Returns:
        str or None: The keyword found in the section title, or None if no keyword is found.
    """
    for keyword in keywords:
        if keyword in section_title.lower():
            return keyword
    return None


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
        input("\nMake relevant search, then press enter").strip().lower()

        # Get the results
        resultater = driver.find_element(By.ID, "resultater")

        # Find all links in the resultater table
        links = resultater.find_elements(By.TAG_NAME, "a")

        # Loop through all links to find the relevant data
        for link in links:
            # Get the URL of the current page
            link_url = link.get_attribute("href")
            # Get the date of the current page
            date_text = link.find_element(By.CLASS_NAME, "col-sm-3").text

            # Click on the link to enter a new web page
            driver.execute_script("window.open();")
            driver.switch_to.window(driver.window_handles[-1])
            link.click()

            # Wait until the new page is loaded
            wait.until(lambda driver: driver.current_url != URL)
            wait.until(lambda driver: driver.find_element(By.TAG_NAME, "body"))
            # Wait until the number of links is greater than 2
            wait.until(lambda driver: len(driver.find_elements(By.TAG_NAME, "a")) > 2)

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

                # Use the identify_keyword function to check for keywords in the section title
                keyword_found = identify_keyword(
                    section_title=section_title, keywords=KEYWORDS
                )
                if keyword_found:
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
                            # Modify the URL to replace "Pdf=false" with "Pdf=true"
                            pdf_url = new_href.replace("Pdf=false", "Pdf=true")

                            # Open the pdf link in a new tab
                            driver.execute_script("window.open();")
                            driver.switch_to.window(driver.window_handles[-1])
                            driver.get(pdf_url)

                            # Wait until the PDF is loaded
                            wait.until(lambda driver: driver.current_url != link_url)

                            # Download the PDF file
                            download_pdf(
                                url=pdf_url,
                                download_dir=DOWNLOAD_DIR + "/" + date_text,
                                new_filename=section_title.replace("/", "_"),
                            )

                            # Close the PDF window/tab and switch back to the original window/tab
                            driver.close()

            driver.close()  # Close this window

    else:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")


if __name__ == "__main__":
    main(url=URL)

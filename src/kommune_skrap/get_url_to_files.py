"""
This module provides functionality to scrape PDF files from a specified URL.
It uses Selenium to interact with the web page and collects file information into a list.
Author:
    Åsmund Frantzen Skomedal
Copyright:
    Åsmund Frantzen Skomedal
License:
    MIT
Constants:
    URL (str): The URL to scrape data from.
    DOWNLOAD_DIR (Path): The directory associated with downloaded PDF files.
    KEYWORDS (list): List of keywords to search for in the scraped data.
Functions:
    identify_keyword(*, section_title, keywords):
        Identify which keyword is present in the section title.
    main(url: str) -> list:
        Use Selenium to scrape kommune data from the given URL.
        Returns a list of tuples (filename, url).
"""

import os
import pickle
from pathlib import Path

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from kommune_skrap.utils import extract_text_from_url

URL = r"https://politiskagenda.kristiansand.kommune.no/"
DOWNLOAD_DIR = Path(r"D:\kommune-skrap\data/kristiansand")
KEYWORDS = ["dispensasjon"]  # Keywords to search for in the scraped data


def main(url: str) -> list:
    """Use selenium to scrape kommune data from the given url.

    Args:
        url (str): URL to scrape data from.

    Returns:
        list: A list of tuples (filename, url) for each file found.
    """
    # Create download directory if it doesn't exist (check if parent directory exists)
    if not os.path.exists(DOWNLOAD_DIR):
        raise FileNotFoundError(f"Directory not found: {DOWNLOAD_DIR}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")
        return

    # Initialize Selenium WebDriver (make sure to have the appropriate driver installed)
    driver = webdriver.Firefox()
    driver.get(URL)

    # Wait until the dropdown is present
    wait = WebDriverWait(driver, 10)

    # Make empty list for all link data
    all_link_data_list = []

    # Prompt the user for input to continue or exit
    go_on = True
    while go_on:
        input_string = (
            input(
                "Make relevant search, then press enter to add links.\n"
                "Write load to load the list from the pickle file.\n"
                'Write "done" when done...\n'
            )
            .strip()
            .lower()
        )
        if input_string == "done":
            go_on = False
        elif input_string == "load":
            # Load the list from the pickle file
            with open(DOWNLOAD_DIR / "all_link_data_list.pkl", "rb") as f:
                all_link_data_list = pickle.load(f)
            print("Loaded the list from the pickle file.")
            go_on = False
        else:
            # Get the results
            resultater = driver.find_element(By.ID, "resultater")

            # Find all links in the resultater table
            links = resultater.find_elements(By.TAG_NAME, "a")
            link_data_list = [
                {
                    "url": link.get_attribute("href"),
                    "date": link.find_element(By.CLASS_NAME, "col-sm-3").text,
                }
                for link in links
            ]
            all_link_data_list.extend(link_data_list)

        # Dump list to a pickle file for later use
        with open(DOWNLOAD_DIR / "all_link_data_list.pkl", "wb") as f:
            pickle.dump(all_link_data_list, f)

    # Make empty list for all file tuples (filename, url)
    files_list = []

    # Prompt the user for input to continue or exit
    for link_data in all_link_data_list:
        link_url = link_data.get("url")
        date_text = link_data.get("date").replace("/", ".")

        # Click on the link to enter a new web page
        driver.execute_script("window.open();")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(link_url)

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

        # Find table called dagsordenDetaljer
        dagsorden_detaljer = driver.find_element(By.ID, "dagsordenDetaljer")

        # Get all the rows in tbody of dagsordenDetaljer
        rows = dagsorden_detaljer.find_elements(By.XPATH, ".//tbody/tr")

        # Loop through all the rows in the table to find the relevant data
        for row in rows:
            # Get the header of the current row
            section_title = row.find_element(
                By.XPATH, ".//h2[contains(@class, 'overskrift')]"
            ).text

            # Use the identify_keyword function to check for keywords in the section title
            # keyword_found = identify_keyword(
            #     section_title=section_title, keywords=KEYWORDS
            # )
            # if keyword_found:
            if True:  # Check all sections, not just those with keywords
                # Get the new links that have appeared after clicking the plus sign
                # Find links that are located in this row of the table
                subpage_links = row.find_elements(By.XPATH, ".//a")
                subpage_links_dict = [
                    {
                        "href": link.get_attribute("href"),
                        "text": link.text,
                    }
                    for link in subpage_links
                ]
                if all(
                    [
                        subpage_link.get("text") == ""
                        for subpage_link in subpage_links_dict
                    ]
                ):
                    continue
                elif not any(
                    [
                        "Link til sak" in subpage_link.get("text")
                        or "Saksfremlegg" in subpage_link.get("text")
                        or "avslag på søknad" in subpage_link.get("text").lower()
                        for subpage_link in subpage_links_dict
                    ]
                ):
                    print(
                        f"\nNOTE: No PDF link found for {section_title}\n"
                        f"on {link_url}\n"
                    )
                # Loop through the new links to find the one that contains the PDF
                for subpage_link in subpage_links_dict:
                    new_href = subpage_link.get("href")
                    if new_href and (
                        "Link til sak" in subpage_link.get("text")
                        or "Saksfremlegg" in subpage_link.get("text")
                        or "avslag på søknad" in subpage_link.get("text").lower()
                    ):
                        try:
                            if new_href.endswith("Pdf=false"):
                                # Modify the URL to replace "Pdf=false" with "Pdf=true"
                                pdf_url = new_href.replace("Pdf=false", "Pdf=true")
                            else:
                                print(
                                    f"\nNOTE: {new_href} does not end with Pdf=false\n"
                                )

                            # Extract text from the PDF to check if it contains the keyword "dispensasjon"
                            pdf_text = extract_text_from_url(pdf_url)
                            keyword_in_pdf = any(
                                keyword.lower() in pdf_text.lower()
                                for keyword in KEYWORDS
                            )

                            # Only proceed if the keyword is found in the PDF
                            if not keyword_in_pdf:
                                print(
                                    f"Skipping {section_title}: keyword not found in PDF"
                                )
                                continue

                            # Set filename
                            new_filename = (
                                section_title.replace("/", "_")
                                .replace("\\", "")
                                .replace('"', "")
                            )
                            # Add number to filename if file already exists
                            count = sum(
                                new_filename in f for f in [f[1] for f in files_list]
                            )
                            if count > 0:
                                new_filename = f"{new_filename}_{count}"

                            # Add the file tuple (filename, url) to the list
                            files_list.append(
                                (date_text, new_filename, pdf_url, link_url)
                            )
                            print(f"Added to list: {new_filename}")
                        except Exception as e:
                            print(f"Failed to process PDF: {new_href}, error: {e}")

        driver.close()  # Close this window
        driver.switch_to.window(driver.window_handles[0])

    return files_list


if __name__ == "__main__":
    files = main(url=URL)
    print(f"\nFound {len(files)} files")
    files_df = pd.DataFrame(
        files, columns=["Dato", "Filnavn", "URL", "Link til dagsorden"]
    )
    files_df.to_csv(DOWNLOAD_DIR / "file_links_2025.csv", index=False)

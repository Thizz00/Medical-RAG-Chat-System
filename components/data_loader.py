import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import wraps

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

from components.config import (
    USER_AGENT,
    REQUEST_TIMEOUT,
    PUBMED_PARAMS,
    PUBMED_SEARCH_URL,
    PUBMED_URL_ARTICLE,
    DRUGS_BASE_URL,
    DRUGS_URL,
)

logger = logging.getLogger(__name__)


@dataclass
class Article:
    title: str
    content: str
    url: str


class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(USER_AGENT)
        self._driver = None

    @property
    def driver(self):
        if not self._driver:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            self._driver = webdriver.Remote(
                command_executor="http://selenium:4444/wd/hub", options=options
            )
        return self._driver

    def __del__(self):
        if self._driver:
            self._driver.quit()

    def fetch_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 403:
                self.driver.get(url)
                return self.driver.page_source
            return response.text if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def parse_article(self, html: str, url: str) -> Dict[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        return {"text": self._extract_text(soup), "metadata": {"source": url}}

    def _extract_text(self, soup: BeautifulSoup) -> str:
        title = soup.find("h1")
        content = soup.find("div", class_=["abstract-content", "ddc-main-content"])
        return (
            f"Title: {title.get_text(strip=True) if title else 'No title'}\n"
            f"Content: {content.get_text(strip=True)[:500] + '...' if content else 'No content'}"
        )


def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return []

    return wrapper


class MedicalDataFetcher:
    def __init__(self):
        self.scraper = WebScraper()

    @handle_exceptions
    def fetch_pubmed(self) -> List[Dict[str, str]]:
        search_url = PUBMED_SEARCH_URL
        params = PUBMED_PARAMS

        response = self.scraper.session.get(search_url, params=params)
        ids = response.json().get("esearchresult", {}).get("idlist", [])

        return [
            self.scraper.parse_article(
                self.scraper.fetch_content(f"{PUBMED_URL_ARTICLE}{id}/"),
                f"{PUBMED_URL_ARTICLE}{id}/",
            )
            for id in ids
            if (html := self.scraper.fetch_content(f"{PUBMED_URL_ARTICLE}{id}/"))
        ]

    @handle_exceptions
    def fetch_drugs(self) -> List[Dict[str, str]]:
        base_url = DRUGS_BASE_URL
        html = self.scraper.fetch_content(base_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = soup.find("ul", class_="ddc-list-column-4").find_all("a", href=True)

        results = []
        for link in links:
            url = f"{DRUGS_URL}{link['href']}"
            if content := self.scraper.fetch_content(url):
                results.append(self.scraper.parse_article(content, url))

        return results


def fetch_medical_data() -> List[Dict[str, str]]:
    logger.info("Fetching medical data")
    fetcher = MedicalDataFetcher()
    data = []

    for fetch_func in [fetcher.fetch_pubmed, fetcher.fetch_drugs]:
        data.extend(fetch_func())

    logger.info(f"Total documents fetched: {len(data)}")
    return data
